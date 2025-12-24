"""
Export a smaller subgraph to JSON for the D3 visualizer.

- Uses the full HeteroData + CSV as source
- Restricts to a subset of users (and their connected posts/comments)
- Deduplicates edges so there is at most ONE edge per (source, target, etype)

Output:
    graph_for_visualizer_small.json
"""

import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

DATA_PT_PATH = "phase1_hetero_gnn_data.pt"
CSV_PATH = "phase1_synthetic_influence_tabular.csv"
OUTPUT_JSON = "graph_for_visualizer_small.json"

# How many users to keep in the visualization graph
MAX_USERS_FOR_VIZ = 2000  # adjust if needed


def node_key(ntype: str, idx: int) -> str:
    return f"{ntype}_{int(idx)}"


def export_small_graph():
    data: HeteroData = torch.load(Path(DATA_PT_PATH))
    df = pd.read_csv(Path(CSV_PATH))

    num_users = int(data["user"].x.size(0))

    # ----- Choose a subset of users -----
    keep_users = list(range(min(MAX_USERS_FOR_VIZ, num_users)))
    keep_users_set = set(keep_users)

    # Map user_id -> influencer_type (only for kept users)
    user_inf_type = {
        int(u): str(t)
        for u, t in zip(df["user_id"].astype(int), df["influencer_type"].astype(str))
        if int(u) in keep_users_set
    }

    # Track which posts/comments are connected to these users
    keep_posts_set = set()
    keep_comments_set = set()

    for (stype, rel, dtype), store in data.edge_items():
        eidx = store.edge_index.cpu().numpy()
        src = eidx[0]
        dst = eidx[1]

        if stype == "user" and dtype == "post":
            # user -> post
            for s, d in zip(src, dst):
                if int(s) in keep_users_set:
                    keep_posts_set.add(int(d))

        if stype == "user" and dtype == "comment":
            # user -> comment
            for s, d in zip(src, dst):
                if int(s) in keep_users_set:
                    keep_comments_set.add(int(d))

        if stype == "comment" and dtype == "post":
            # comment -> post: if we keep the comment, keep that post too
            for s, d in zip(src, dst):
                if int(s) in keep_comments_set:
                    keep_posts_set.add(int(d))

    # ----- Build nodes -----
    nodes_dict = {}

    # Users
    for u in keep_users:
        nid = node_key("user", u)
        nodes_dict[nid] = {
            "id": nid,
            "ntype": "user",
            "index": int(u),
            "influencer_type": user_inf_type.get(u, "none"),
        }

    # Posts
    num_posts = int(data["post"].x.size(0))
    for p in keep_posts_set:
        if 0 <= p < num_posts:
            nid = node_key("post", p)
            nodes_dict[nid] = {
                "id": nid,
                "ntype": "post",
                "index": int(p),
            }

    # Comments
    num_comments = int(data["comment"].x.size(0))
    for c in keep_comments_set:
        if 0 <= c < num_comments:
            nid = node_key("comment", c)
            nodes_dict[nid] = {
                "id": nid,
                "ntype": "comment",
                "index": int(c),
            }

    # ----- Build edges with de-duplication -----
    edges_list = []
    edge_seen = set()  # (source, target, etype)

    for (stype, rel, dtype), store in data.edge_items():
        eidx = store.edge_index.cpu().numpy()
        src_arr = eidx[0]
        dst_arr = eidx[1]

        for s, d in zip(src_arr, dst_arr):
            s = int(s)
            d = int(d)

            # Only keep edges where the src user is in our subset,
            # or src node (post/comment) is also in nodes_dict.
            if stype == "user" and s not in keep_users_set:
                continue
            if stype == "post" and node_key("post", s) not in nodes_dict:
                continue
            if stype == "comment" and node_key("comment", s) not in nodes_dict:
                continue

            s_name = node_key(stype, s)
            d_name = node_key(dtype, d)

            if s_name not in nodes_dict or d_name not in nodes_dict:
                continue

            edge_key = (s_name, d_name, str(rel))
            if edge_key in edge_seen:
                # skip duplicate edge (same source, target, etype)
                continue
            edge_seen.add(edge_key)

            edges_list.append(
                {
                    "source": s_name,
                    "target": d_name,
                    "etype": str(rel),
                }
            )

    graph_json = {
        "nodes": list(nodes_dict.values()),
        "edges": edges_list,
    }

    out_path = Path(OUTPUT_JSON)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(graph_json, f)

    print(f"[INFO] Exported SMALL graph to: {out_path.resolve()}")
    print(f"[INFO] #nodes: {len(nodes_dict)}, #edges: {len(edges_list)}")
    print("[INFO] Use this file in the HTML visualizer: graph_for_visualizer_small.json")


if __name__ == "__main__":
    export_small_graph()
