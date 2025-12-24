"""
Find posts in graph_for_visualizer_small.json
that have many incoming edges from different users.
"""

import json
from pathlib import Path
from collections import defaultdict

JSON_PATH = "graph_for_visualizer_medium.json"
TOP_K = 10

def main():
    data = json.loads(Path(JSON_PATH).read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    edges = data["edges"]

    # post_id -> set(user_ids)
    post_to_users = defaultdict(set)

    for e in edges:
        s = e["source"]
        t = e["target"]
        et = e["etype"]

        if t.startswith("post_"):
            # store only user sources
            if s.startswith("user_"):
                post_to_users[t].add(s)
            # also: comments under post -> users who wrote comment
            if s.startswith("comment_"):
                # comment -> post (under) we'll handle indirectly if you want, but
                # this already captures user->post edges for posted/liked/reshared.
                pass

    scored = [(p, len(users)) for p, users in post_to_users.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"[INFO] Found {len(scored)} posts.")
    print(f"[INFO] Top {TOP_K} intersection posts in SMALL graph:")
    for i, (p, cnt) in enumerate(scored[:TOP_K], start=1):
        example_users = sorted(list(post_to_users[p]))[:10]
        print(f"{i:2d}) {p}, distinct_users = {cnt}, example_users = {example_users}")

    if scored:
        best_post, _ = scored[0]
        print("\n[SUGGESTION] For visualizer, use intersection node:", best_post)

if __name__ == "__main__":
    main()
