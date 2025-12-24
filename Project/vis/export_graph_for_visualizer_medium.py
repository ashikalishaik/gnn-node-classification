"""
Export a medium-sized subgraph to JSON for the D3 visualizer.

Goal:
  - Bigger than the very small sample.
  - Much smaller than the full 1.2M / 8.8M graph.
  - Every visible post has a visible creator.

Output:
  graph_for_visualizer_medium.json
"""

import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

DATA_PT_PATH = "phase1_hetero_gnn_data.pt"
CSV_PATH = "phase1_synthetic_influence_tabular.csv"
OUTPUT_JSON = "graph_for_visualizer_medium.json"

# ----- SIZE TUNING -----
# Start with this many users by index: user_0, user_1, ..., user_(BASE_MAX_USERS-1)
# You can increase this slowly: 2000 → 5000 → 10000, etc.
BASE_MAX_USERS = 8000


def node_key(ntype: str, idx: int) -> str:
    return f"{ntype}_{int(idx)}"


def export_medium_graph():
    data: HeteroData = torch.load(Path(DATA_PT_PATH))
    df = pd.read_csv(Path(CSV_PATH))

    num_users = int(data["user"].x.size(0))
    print(f"[INFO] Total users in full graph: {num_users}")

    # ----- 1) Base user sample -----
    base_users = list(range(min(BASE_MAX_USERS, num_users)))
    keep_users_set = set(base_users)

    # Map user_id -> influencer_type (for labels, only for kept users)
    user_inf_type = {
        int(u): str(t)
        for u, t in zip(df["user_id"].astype(int), df["influencer_type"].astype(str))
        if int(u) in keep_users_set
    }

    # Short alias for edges
    def get_edge(etype):
        if etype not in data.edge_types:
            return None
        return data[etype].edge_index.cpu().numpy()

    e_posted    = get_edge(("user", "posted",   "post"))      # user -> post
    e_liked     = get_edge(("user", "liked",    "post"))      # user -> post
    e_reshared  = get_edge(("user", "reshared", "post"))      # user -> post
    e_u_comment = get_edge(("user", "commented","comment"))   # user -> comment
    e_c_under   = get_edge(("comment","under",  "post"))      # comment -> post

    # ----- 2) From those users, collect connected posts & comments -----
    keep_posts_set = set()
    keep_comments_set = set()

    # (a) posts they posted / liked / reshared
    def add_user_post_edges(eidx):
        if eidx is None:
            return
        src, dst = eidx
        for u, p in zip(src, dst):
            u = int(u)
            p = int(p)
            if u in keep_users_set:
                keep_posts_set.add(p)

    add_user_post_edges(e_posted)
    add_user_post_edges(e_liked)
    add_user_post_edges(e_reshared)

    # (b) comments they wrote
    if e_u_comment is not None:
        src, dst = e_u_comment
        for u, c in zip(src, dst):
            u = int(u)
            c = int(c)
            if u in keep_users_set:
                keep_comments_set.add(c)

    # (c) posts those comments are under
    comment_to_post = {}
    if e_c_under is not None:
        c_src, p_dst = e_c_under
        for c, p in zip(c_src, p_dst):
            c = int(c)
            p = int(p)
            comment_to_post[c] = p
            if c in keep_comments_set:
                keep_posts_set.add(p)

    print(f"[INFO] After base sampling:")
    print(f"  kept users (base): {len(keep_users_set)}")
    print(f"  connected posts   : {len(keep_posts_set)}")
    print(f"  connected comments: {len(keep_comments_set)}")

    # ----- 3) Ensure: every kept post has its creator user also kept -----
    if e_posted is not None:
        src, dst = e_posted
        for u, p in zip(src, dst):
            u = int(u)
            p = int(p)
            if p in keep_posts_set:
                keep_users_set.add(u)

    print(f"[INFO] After adding post creators:")
    print(f"  kept users (with creators): {len(keep_users_set)}")

    # Refresh influencer_type mapping for any newly added users
    user_inf_type = {
        int(u): str(t)
        for u, t in zip(df["user_id"].astype(int), df["influencer_type"].astype(str))
        if int(u) in keep_users_set
    }

    # ----- 4) Build node dict -----
    nodes_dict = {}

    # Users
    for u in keep_users_set:
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

    # ----- 5) Build edges (no duplicates) -----
    edges_list = []
    edge_seen = set()  # (source, target, etype)

    for (stype, rel, dtype), store in data.edge_items():
        eidx = store.edge_index.cpu().numpy()
        src_arr = eidx[0]
        dst_arr = eidx[1]
        rel_str = str(rel)

        for s, d in zip(src_arr, dst_arr):
            s = int(s)
            d = int(d)
            s_name = node_key(stype, s)
            d_name = node_key(dtype, d)

            if s_name not in nodes_dict or d_name not in nodes_dict:
                continue

            edge_key = (s_name, d_name, rel_str)
            if edge_key in edge_seen:
                continue
            edge_seen.add(edge_key)

            edges_list.append(
                {
                    "source": s_name,
                    "target": d_name,
                    "etype": rel_str,
                }
            )

    graph_json = {
        "nodes": list(nodes_dict.values()),
        "edges": edges_list,
    }

    out_path = Path(OUTPUT_JSON)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(graph_json, f)

    print(f"[INFO] Exported MEDIUM graph to: {out_path.resolve()}")
    print(f"[INFO] #nodes: {len(nodes_dict)}, #edges: {len(edges_list)}")
    print("[INFO] Use this file in the HTML visualizer: graph_for_visualizer_medium.json")


if __name__ == "__main__":
    export_medium_graph()
