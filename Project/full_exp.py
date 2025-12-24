"""
GNN vs XGBoost vs MLP on synthetic "influencer / influencer-of-influencers" graph.

Goal:
- Show that GNNs can correctly identify "influencers of influencers" (meta influencers)
  by using multi-hop graph structure, while XGBoost and MLP (tabular models) struggle
  because they only see local scalar features (num_followers, num_following, etc.).

Classes:
  0 = normal user
  1 = influencer (many direct followers)
  2 = meta influencer (influencers-of-influencers)
      - moderate follower counts (similar to normal users)
      - but *many of their followers are influencers* (multi-hop signal)

Models:
  - XGBoost (tabular)
  - MLP (tabular)
  - GNN (GraphSAGE) on user-user follower graph

Outputs:
  - Prints per-model metrics (accuracy, macro-F1)
  - Prints per-class precision/recall/F1, highlighting class 2 (meta)
  - Saves simple confusion matrix plots to ./results/ (optional, for your slides)
"""

import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ============================================================
# 0. Global config
# ============================================================

RNG_SEED = 42
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

def set_seed(seed: int = RNG_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. Synthetic graph generator
# ============================================================

def generate_influence_graph(
    num_normal: int = 6000,
    num_influencers: int = 3000,
    num_meta: int = 1000,
):
    """
    Build a *user-only* directed follower graph:
      edge (u -> v) means "u follows v".

    Design:
    - Influencers (class 1):
        - Very high in-degree (many followers, mostly normals).
    - Meta influencers (class 2):
        - Moderate in-degree (similar to normal users).
        - BUT a large fraction of their followers are influencers.
          => multi-hop signal to detect them.
    - Normal users (class 0):
        - Mix of low and some moderate followers.
        - Some normals will intentionally have similar follower counts
          to meta, but mainly from other normals (NOT from influencers),
          to confuse tabular models.

    Node features:
      x[i] = [num_followers, num_following, posts, comments]
      (same features are used for XGBoost/MLP and as GNN node features)
    """

    set_seed(RNG_SEED)

    total_nodes = num_normal + num_influencers + num_meta
    node_ids = np.arange(total_nodes)

    # Assign class segments
    normal_ids = node_ids[0:num_normal]
    infl_ids = node_ids[num_normal:num_normal + num_influencers]
    meta_ids = node_ids[num_normal + num_influencers:]

    labels = np.zeros(total_nodes, dtype=int)
    labels[infl_ids] = 1
    labels[meta_ids] = 2

    # Followers/following
    num_followers = np.zeros(total_nodes, dtype=int)
    num_following = np.zeros(total_nodes, dtype=int)

    # Posts/comments base distribution: similar across all classes
    base_posts = np.random.poisson(lam=120, size=total_nodes)
    base_comments = np.random.poisson(lam=400, size=total_nodes)

    post_scale = np.ones(total_nodes)
    comment_scale = np.ones(total_nodes)

    # Slightly more active influencers/meta, but overlapping
    post_scale[infl_ids] = np.random.normal(loc=1.05, scale=0.05, size=len(infl_ids))
    post_scale[meta_ids] = np.random.normal(loc=1.05, scale=0.05, size=len(meta_ids))

    comment_scale[infl_ids] = np.random.normal(loc=1.05, scale=0.05, size=len(infl_ids))
    comment_scale[meta_ids] = np.random.normal(loc=1.05, scale=0.05, size=len(meta_ids))

    num_posts = (base_posts * post_scale).astype(int)
    num_comments = (base_comments * comment_scale).astype(int)
    num_posts = np.clip(num_posts, 0, None)
    num_comments = np.clip(num_comments, 0, None)

    # ------------------------------------------------------------------
    # Build follower edges (u -> v means u follows v)
    # ------------------------------------------------------------------
    src_edges = []
    dst_edges = []

    # Helper to add edges (avoid exact duplicates with a set)
    edge_set = set()

    def add_edge(u, v):
        if u == v:
            return
        key = (int(u), int(v))
        if key in edge_set:
            return
        edge_set.add(key)
        src_edges.append(int(u))
        dst_edges.append(int(v))

    # 1) Normal users: follow mostly influencers + some normals/meta
    for u in normal_ids:
        # each normal follows ~20 accounts
        k = np.random.poisson(lam=20)
        k = max(5, k)  # ensure at least 5

        # 60% influencers, 30% normals, 10% meta
        n_inf = int(k * 0.6)
        n_norm = int(k * 0.3)
        n_meta = k - n_inf - n_norm

        if len(infl_ids) > 0 and n_inf > 0:
            v_inf = np.random.choice(infl_ids, size=min(n_inf, len(infl_ids)), replace=True)
            for v in v_inf:
                add_edge(u, v)

        if len(normal_ids) > 1 and n_norm > 0:
            v_norm = np.random.choice(normal_ids, size=min(n_norm, len(normal_ids)), replace=True)
            for v in v_norm:
                add_edge(u, v)

        if len(meta_ids) > 0 and n_meta > 0:
            v_meta = np.random.choice(meta_ids, size=min(n_meta, len(meta_ids)), replace=True)
            for v in v_meta:
                add_edge(u, v)

    # 2) Influencers: have many followers from normals, and follow metas (to create "meta" structure)
    for v in infl_ids:
        # Many normal followers
        k_followers = np.random.poisson(lam=150)
        followers = np.random.choice(normal_ids, size=min(k_followers, len(normal_ids)), replace=True)
        for u in followers:
            add_edge(u, v)

        # Influencers might follow some metas (so meta gets influencer followers)
        k_meta = np.random.poisson(lam=8)
        if len(meta_ids) > 0 and k_meta > 0:
            metas = np.random.choice(meta_ids, size=min(k_meta, len(meta_ids)), replace=True)
            for m in metas:
                add_edge(v, m)

        # Influencers might also follow each other sparsely
        k_inf_follow = np.random.poisson(lam=5)
        others = np.random.choice(infl_ids, size=min(k_inf_follow, len(infl_ids)), replace=True)
        for o in others:
            add_edge(v, o)

    # 3) Meta influencers:
    #    We want their followers to be a mix of influencers and normals,
    #    with a *large proportion* of influencers (multi-hop signal).
    for m in meta_ids:
        # Many influencer followers
        k_inf_followers = np.random.poisson(lam=60)
        if len(infl_ids) > 0 and k_inf_followers > 0:
            inf_followers = np.random.choice(infl_ids, size=min(k_inf_followers, len(infl_ids)), replace=True)
            for u in inf_followers:
                add_edge(u, m)

        # Some normal followers, to keep follower counts similar to some normals
        k_norm_followers = np.random.poisson(lam=40)
        if len(normal_ids) > 0 and k_norm_followers > 0:
            norm_followers = np.random.choice(normal_ids, size=min(k_norm_followers, len(normal_ids)), replace=True)
            for u in norm_followers:
                add_edge(u, m)

        # Meta may follow several influencers as well
        k_follow_inf = np.random.poisson(lam=20)
        if len(infl_ids) > 0 and k_follow_inf > 0:
            inf_targets = np.random.choice(infl_ids, size=min(k_follow_inf, len(infl_ids)), replace=True)
            for v in inf_targets:
                add_edge(m, v)

    # Now compute final followers/following from edges
    src_edges_arr = np.array(src_edges, dtype=int)
    dst_edges_arr = np.array(dst_edges, dtype=int)

    for u in range(total_nodes):
        num_following[u] = int((src_edges_arr == u).sum())
        num_followers[u] = int((dst_edges_arr == u).sum())

    # Build node features
    x = np.stack(
        [
            num_followers.astype(np.float32),
            num_following.astype(np.float32),
            num_posts.astype(np.float32),
            num_comments.astype(np.float32),
        ],
        axis=1,
    )  # shape: (N, 4)

    # Torch Geometric Data object (homogeneous graph)
    edge_index = torch.tensor(
        np.vstack([src_edges_arr, dst_edges_arr]), dtype=torch.long
    )
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x_t, edge_index=edge_index, y=y_t)

    # Also return numpy arrays for tabular baselines
    return data, x, labels


# ============================================================
# 2. Train/val/test split
# ============================================================

def make_splits(labels, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    """
    Per-class splits with given fractions. Returns boolean masks.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    labels = np.array(labels)
    num_nodes = len(labels)

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        np.random.shuffle(idx)
        n = len(idx)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        # rest goes to test

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


# ============================================================
# 3. XGBoost baseline (tabular)
# ============================================================

def train_eval_xgboost(x, labels, train_mask, val_mask, test_mask):
    print("\n=== XGBoost (tabular) ===")

    X_train = x[train_mask]
    y_train = labels[train_mask]

    X_val = x[val_mask]
    y_val = labels[val_mask]

    X_test = x[test_mask]
    y_test = labels[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=RNG_SEED,
    )

    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)

    y_pred = model.predict(X_test_s)

    return evaluate_classifier("XGBoost", y_test, y_pred)


# ============================================================
# 4. MLP baseline (tabular)
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_eval_mlp(x, labels, train_mask, val_mask, test_mask, device=None):
    print("\n=== MLP (tabular) ===")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = x[train_mask]
    y_train = labels[train_mask]

    X_val = x[val_mask]
    y_val = labels[val_mask]

    X_test = x[test_mask]
    y_test = labels[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    model = MLP(in_dim=X_train.shape[1], hidden_dim=64, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, 61):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        # val
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_true = y_val_t.cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: train_loss={loss.item():.4f}, val_macroF1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_pred = test_logits.argmax(dim=1).cpu().numpy()

    y_test_np = y_test_t.cpu().numpy()
    return evaluate_classifier("MLP", y_test_np, test_pred)


# ============================================================
# 5. GNN (GraphSAGE) model
# ============================================================

class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lin(x)
        return x


def train_eval_gnn(data: Data, train_mask, val_mask, test_mask, device=None):
    print("\n=== GNN (GraphSAGE) ===")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = data.to(device)
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool, device=device)
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool, device=device)

    num_features = data.x.size(1)
    model = GraphSAGEModel(in_dim=num_features, hidden_dim=64, num_classes=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, 81):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask_t], data.y[train_mask_t])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)[val_mask_t]
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_true = data.y[val_mask_t].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: train_loss={loss.item():.4f}, val_macroF1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)[test_mask_t]
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = data.y[test_mask_t].cpu().numpy()

    return evaluate_classifier("GNN", y_true, y_pred)


# ============================================================
# 6. Evaluation helper
# ============================================================

def evaluate_classifier(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # Per-class
    per_class = {}
    for cls in [0, 1, 2]:
        mask = (y_true == cls)
        if mask.sum() == 0:
            continue
        per_class[cls] = {
            "support": int(mask.sum()),
            "precision": float(precision_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)),
        }

    print(f"\n{name} – Overall:")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")

    print(f"\n{name} – Per-class (0=normal, 1=influencer, 2=meta):")
    for cls, m in per_class.items():
        print(
            f"  Class {cls}: "
            f"support={m['support']}, "
            f"precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}, "
            f"f1={m['f1']:.4f}"
        )

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name} – Confusion Matrix")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["normal", "influencer", "meta"], rotation=45, ha="right")
    ax.set_yticklabels(["normal", "influencer", "meta"])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha="center", va="center", color="black")
    fig.tight_layout()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = RESULT_DIR / f"cm_{name.lower().replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "cm_path": str(out_path),
    }


# ============================================================
# 7. Main runner
# ============================================================

def main():
    print("Generating synthetic influence graph...")
    data, x_np, labels_np = generate_influence_graph()
    print("Graph:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")

    # Splits
    train_mask, val_mask, test_mask = make_splits(labels_np, 0.6, 0.2, 0.2)
    data.train_mask = torch.tensor(train_mask)
    data.val_mask = torch.tensor(val_mask)
    data.test_mask = torch.tensor(test_mask)

    # Run baselines & GNN
    xgb_metrics = train_eval_xgboost(x_np, labels_np, train_mask, val_mask, test_mask)
    mlp_metrics = train_eval_mlp(x_np, labels_np, train_mask, val_mask, test_mask)
    gnn_metrics = train_eval_gnn(data, train_mask, val_mask, test_mask)

    # Summary comparison
    print("\n\n================= FINAL COMPARISON =================")
    for m in [xgb_metrics, mlp_metrics, gnn_metrics]:
        print(f"\n{m['name']}:")
        print(f"  Accuracy : {m['accuracy']:.4f}")
        print(f"  Macro F1 : {m['macro_f1']:.4f}")
        meta_stats = m["per_class"].get(2, None)
        if meta_stats is not None:
            print(
                f"  Class 2 (meta) – "
                f"precision={meta_stats['precision']:.4f}, "
                f"recall={meta_stats['recall']:.4f}, "
                f"f1={meta_stats['f1']:.4f}"
            )
        print(f"  Confusion matrix image: {m['cm_path']}")

    print("\nNote: The key thing to check is Class 2 (meta influencers).")
    print("GNN should achieve significantly higher recall/F1 on meta than XGBoost/MLP,")
    print("because it uses neighbors' features (multi-hop) to see that meta nodes")
    print("are followed by many influencer nodes, even though their own follower")
    print("counts look similar to some normal users.")


if __name__ == "__main__":
    main()
