"""
Binary task: (normal) vs (influencer + meta influencer)

This script:
  - Generates a synthetic follower graph with 3 internal types:
        0 = normal
        1 = influencer
        2 = meta influencer (influencer-of-influencers)
  - Defines a binary label:
        y_bin = 0 for normal
        y_bin = 1 for influencer or meta
  - Builds tabular features where normals and meta have very similar stats.
  - Trains:
        * XGBoost (tabular)
        * MLP (tabular)
        * GNN (GraphSAGE, full-graph)
  - Computes and SAVES:
        * Confusion matrices for each model
        * ROC curves + AUC for each model
        * MLP training curves (train/val accuracy, val macro F1)
        * GNN training curves (train/val accuracy, val macro F1)
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


# ============================================================
# 0. Global config
# ============================================================

RNG_SEED = 42
RESULT_DIR = Path("results_binary")
RESULT_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = RNG_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. Synthetic graph generator (binary setting)
# ============================================================

def generate_influence_graph_binary(
    num_normal: int = 6000,
    num_influencers: int = 3000,
    num_meta: int = 1000,
):
    """
    Build a user-only directed follower graph:
      edge (u -> v) means "u follows v".

    Internal type labels (type3):
      0 = normal (many)
      1 = influencer (large accounts)
      2 = meta influencer (influencers-of-influencers)

    Binary training label:
      y_bin = 0 for normal
      y_bin = 1 for influencer or meta

    Features (intentionally similar for normal & meta):
      x[i] = [num_followers_feat, num_following_feat, num_posts, num_comments]
    """

    set_seed(RNG_SEED)

    total_nodes = num_normal + num_influencers + num_meta
    node_ids = np.arange(total_nodes)

    # Assign type segments
    normal_ids = node_ids[:num_normal]
    infl_ids = node_ids[num_normal:num_normal + num_influencers]
    meta_ids = node_ids[num_normal + num_influencers:]

    labels_3 = np.zeros(total_nodes, dtype=int)
    labels_3[infl_ids] = 1
    labels_3[meta_ids] = 2

    # Binary label
    labels_bin = (labels_3 != 0).astype(int)

    # ------------------------------------------------------------------
    # Graph structure: who follows whom  (u -> v means "u follows v")
    # ------------------------------------------------------------------
    src_edges = []
    dst_edges = []
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
        k = np.random.poisson(lam=20)
        k = max(5, k)

        n_inf = int(k * 0.6)
        n_norm = int(k * 0.3)
        n_meta = k - n_inf - n_norm

        if len(infl_ids) > 0 and n_inf > 0:
            v_inf = np.random.choice(infl_ids, size=n_inf, replace=True)
            for v in v_inf:
                add_edge(u, v)

        if len(normal_ids) > 1 and n_norm > 0:
            v_norm = np.random.choice(normal_ids, size=n_norm, replace=True)
            for v in v_norm:
                add_edge(u, v)

        if len(meta_ids) > 0 and n_meta > 0:
            v_meta = np.random.choice(meta_ids, size=n_meta, replace=True)
            for v in v_meta:
                add_edge(u, v)

    # 2) Influencers:
    #    - Already have followers from normals (edges created above).
    #    - Also follow metas and other influencers.
    for v in infl_ids:
        # influencers follow some metas
        k_meta = np.random.poisson(lam=10)
        if len(meta_ids) > 0 and k_meta > 0:
            metas = np.random.choice(meta_ids, size=k_meta, replace=True)
            for m in metas:
                add_edge(v, m)

        # influencers follow some other influencers
        k_inf_follow = np.random.poisson(lam=5)
        if len(infl_ids) > 1 and k_inf_follow > 0:
            others = np.random.choice(infl_ids, size=k_inf_follow, replace=True)
            for o in others:
                add_edge(v, o)

    # 3) Meta influencers:
    #    - Many influencer followers + some normal followers.
    #    - Follow some influencers as well.
    for m in meta_ids:
        # influencer followers
        k_inf_followers = np.random.poisson(lam=60)
        if len(infl_ids) > 0 and k_inf_followers > 0:
            inf_followers = np.random.choice(infl_ids, size=k_inf_followers, replace=True)
            for u in inf_followers:
                add_edge(u, m)

        # normal followers
        k_norm_followers = np.random.poisson(lam=40)
        if len(normal_ids) > 0 and k_norm_followers > 0:
            norm_followers = np.random.choice(normal_ids, size=k_norm_followers, replace=True)
            for u in norm_followers:
                add_edge(u, m)

        # meta follow some influencers
        k_follow_inf = np.random.poisson(lam=20)
        if len(infl_ids) > 0 and k_follow_inf > 0:
            inf_targets = np.random.choice(infl_ids, size=k_follow_inf, replace=True)
            for v in inf_targets:
                add_edge(m, v)

    src_edges_arr = np.array(src_edges, dtype=int)
    dst_edges_arr = np.array(dst_edges, dtype=int)

    # ------------------------------------------------------------------
    # Tabular features – intentionally similar for normals & meta
    # ------------------------------------------------------------------
    num_followers_feat = np.zeros(total_nodes, dtype=float)
    num_following_feat = np.zeros(total_nodes, dtype=float)

    # Normals & meta share almost the same feature distributions
    mu_nm_followers = 1200.0
    sigma_nm_followers = 200.0
    mu_nm_following = 600.0
    sigma_nm_following = 100.0

    # Normals
    num_followers_feat[normal_ids] = np.random.normal(
        loc=mu_nm_followers, scale=sigma_nm_followers, size=len(normal_ids)
    )
    num_following_feat[normal_ids] = np.random.normal(
        loc=mu_nm_following, scale=sigma_nm_following, size=len(normal_ids)
    )

    # Meta (slightly scaled, but overlapping heavily with normals)
    num_followers_feat[meta_ids] = np.random.normal(
        loc=mu_nm_followers * 1.02, scale=sigma_nm_followers, size=len(meta_ids)
    )
    num_following_feat[meta_ids] = np.random.normal(
        loc=mu_nm_following * 1.02, scale=sigma_nm_following, size=len(meta_ids)
    )

    # Influencers: clearly larger
    num_followers_feat[infl_ids] = np.random.normal(
        loc=20_000.0, scale=3_000.0, size=len(infl_ids)
    )
    num_following_feat[infl_ids] = np.random.normal(
        loc=2_000.0, scale=400.0, size=len(infl_ids)
    )

    # Posts & comments
    base_posts = np.random.poisson(lam=120, size=total_nodes).astype(float)
    base_comments = np.random.poisson(lam=400, size=total_nodes).astype(float)

    post_scale = np.ones(total_nodes)
    comment_scale = np.ones(total_nodes)

    post_scale[infl_ids] *= np.random.normal(loc=1.05, scale=0.05, size=len(infl_ids))
    post_scale[meta_ids] *= np.random.normal(loc=1.03, scale=0.05, size=len(meta_ids))

    comment_scale[infl_ids] *= np.random.normal(loc=1.05, scale=0.05, size=len(infl_ids))
    comment_scale[meta_ids] *= np.random.normal(loc=1.03, scale=0.05, size=len(meta_ids))

    num_posts = (base_posts * post_scale).clip(min=0)
    num_comments = (base_comments * comment_scale).clip(min=0)

    num_followers_feat = np.clip(num_followers_feat, 0, None)
    num_following_feat = np.clip(num_following_feat, 0, None)

    x = np.stack(
        [
            num_followers_feat.astype(np.float32),
            num_following_feat.astype(np.float32),
            num_posts.astype(np.float32),
            num_comments.astype(np.float32),
        ],
        axis=1,
    )

    edge_index = torch.tensor(
        np.vstack([src_edges_arr, dst_edges_arr]), dtype=torch.long
    )
    x_t = torch.tensor(x, dtype=torch.float32)
    y_bin_t = torch.tensor(labels_bin, dtype=torch.long)
    data = Data(x=x_t, edge_index=edge_index, y=y_bin_t)

    return data, x, labels_bin, labels_3


# ============================================================
# 2. Train/val/test split (stratified on 3-class type)
# ============================================================

def make_splits(labels_3, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    labels_3 = np.array(labels_3)
    num_nodes = len(labels_3)

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    for cls in [0, 1, 2]:
        idx = np.where(labels_3 == cls)[0]
        np.random.shuffle(idx)
        n = len(idx)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


# ============================================================
# 3. Evaluation helper (binary, with ROC/AUC)
# ============================================================

def evaluate_binary(name, y_true_bin, y_pred_bin, y_proba, labels_3_test):
    """
    y_true_bin, y_pred_bin: arrays of 0/1 for test nodes
    y_proba: predicted probabilities for class 1
    labels_3_test: original 3-class labels for test nodes (0/1/2)
    """
    acc = accuracy_score(y_true_bin, y_pred_bin)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro")
    prec_pos = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    rec_pos = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    f1_pos = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

    # Meta-subset metrics
    meta_mask = (labels_3_test == 2)
    y_meta_true = y_true_bin[meta_mask]
    y_meta_pred = y_pred_bin[meta_mask]

    if meta_mask.sum() > 0:
        meta_prec = precision_score(y_meta_true, y_meta_pred, pos_label=1, zero_division=0)
        meta_rec = recall_score(y_meta_true, y_meta_pred, pos_label=1, zero_division=0)
        meta_f1 = f1_score(y_meta_true, y_meta_pred, pos_label=1, zero_division=0)
    else:
        meta_prec = meta_rec = meta_f1 = 0.0

    # ROC + AUC
    fpr, tpr, _ = roc_curve(y_true_bin, y_proba)
    auc_value = roc_auc_score(y_true_bin, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title(f"{name} – Binary Confusion Matrix")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["normal", "infl/meta"], rotation=45, ha="right")
    ax_cm.set_yticklabels(["normal", "infl/meta"])
    for (i, j), val in np.ndenumerate(cm):
        ax_cm.text(j, i, val, ha="center", va="center", color="black")
    fig_cm.tight_layout()
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cm_path = RESULT_DIR / f"cm_binary_{name.lower().replace(' ', '_')}.png"
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)

    # ROC curve plot
    fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")
    ax_roc.plot([0, 1], [0, 1], "--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{name} – ROC Curve")
    ax_roc.legend()
    ax_roc.grid(True)
    roc_path = RESULT_DIR / f"roc_binary_{name.lower().replace(' ', '_')}.png"
    fig_roc.savefig(roc_path, dpi=150)
    plt.close(fig_roc)

    # Print summary
    print(f"\n{name} – Overall binary:")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Macro F1         : {macro_f1:.4f}")
    print(f"  Pos-class Prec   : {prec_pos:.4f}")
    print(f"  Pos-class Recall : {rec_pos:.4f}")
    print(f"  Pos-class F1     : {f1_pos:.4f}")
    print(f"  AUC              : {auc_value:.4f}")

    print(f"\n{name} – Meta (type3=2) subset:")
    print(f"  # Meta test nodes: {meta_mask.sum()}")
    print(f"  Precision (meta) : {meta_prec:.4f}")
    print(f"  Recall (meta)    : {meta_rec:.4f}")
    print(f"  F1 (meta)        : {meta_f1:.4f}")
    print(f"  Confusion matrix : {cm_path}")
    print(f"  ROC curve        : {roc_path}")

    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "pos_precision": prec_pos,
        "pos_recall": rec_pos,
        "pos_f1": f1_pos,
        "meta_precision": meta_prec,
        "meta_recall": meta_rec,
        "meta_f1": meta_f1,
        "auc": auc_value,
        "cm_path": str(cm_path),
        "roc_path": str(roc_path),
    }


# ============================================================
# 4. XGBoost baseline (binary)
# ============================================================

def run_xgboost(x_np, y_bin, labels_3, train_mask, val_mask, test_mask):
    print("\n=== XGBoost (binary) ===")

    X_train = x_np[train_mask]
    y_train = y_bin[train_mask]

    X_val = x_np[val_mask]
    y_val = y_bin[val_mask]

    X_test = x_np[test_mask]
    y_test = y_bin[test_mask]
    labels_3_test = labels_3[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=RNG_SEED,
    )

    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)

    y_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return evaluate_binary("XGBoost", y_test, y_pred, y_proba, labels_3_test)


# ============================================================
# 5. MLP baseline (binary, with training curves)
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),  # binary logit
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_mlp(x_np, y_bin, labels_3, train_mask, val_mask, test_mask, device=None):
    print("\n=== MLP (binary) ===")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = x_np[train_mask]
    y_train = y_bin[train_mask]

    X_val = x_np[val_mask]
    y_val = y_bin[val_mask]

    X_test = x_np[test_mask]
    y_test = y_bin[test_mask]
    labels_3_test = labels_3[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    model = MLP(in_dim=X_train.shape[1], hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_state = None

    # For training curves
    epochs = 60
    epoch_list = []
    train_acc_hist = []
    val_acc_hist = []
    val_f1_hist = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        # Train accuracy
        with torch.no_grad():
            train_proba = torch.sigmoid(logits).cpu().numpy()
            train_pred = (train_proba >= 0.5).astype(int)
            train_true = y_train_t.cpu().numpy()
            train_acc = accuracy_score(train_true, train_pred)

        # Val metrics
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_proba = torch.sigmoid(val_logits).cpu().numpy()
            val_pred = (val_proba >= 0.5).astype(int)
            val_true = y_val_t.cpu().numpy()
            val_acc = accuracy_score(val_true, val_pred)
            val_f1 = f1_score(val_true, val_pred, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

        epoch_list.append(epoch)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        val_f1_hist.append(val_f1)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}: train_loss={loss.item():.4f}, "
                f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_macroF1={val_f1:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save training curve plot (accuracy + val F1)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epoch_list, train_acc_hist, label="Train Acc")
    ax1.plot(epoch_list, val_acc_hist, label="Val Acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(epoch_list, val_f1_hist, label="Val Macro F1", linestyle="--")
    ax2.set_ylabel("Val Macro F1")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    curve_path = RESULT_DIR / "mlp_training_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] MLP training curve saved to: {curve_path}")

    # Test
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_proba = torch.sigmoid(test_logits).cpu().numpy()
        test_pred = (test_proba >= 0.5).astype(int)

    y_test_np = y_test_t.cpu().numpy()
    return evaluate_binary("MLP", y_test_np, test_pred, test_proba, labels_3_test)


# ============================================================
# 6. GNN (GraphSAGE, binary, with training curves)
# ============================================================

class GraphSAGEBinary(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lin(x).squeeze(-1)
        return x


def run_gnn(data: Data, labels_3, train_mask, val_mask, test_mask, device=None):
    print("\n=== GNN (GraphSAGE, binary) ===")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = data.to(device)
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool, device=device)
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool, device=device)

    num_features = data.x.size(1)
    model = GraphSAGEBinary(in_dim=num_features, hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_state = None

    # For training curves
    epochs = 80
    epoch_list = []
    train_acc_hist = []
    val_acc_hist = []
    val_f1_hist = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask_t], data.y[train_mask_t].float())
        loss.backward()
        optimizer.step()

        # Train accuracy
        with torch.no_grad():
            train_logits = logits[train_mask_t]
            train_proba = torch.sigmoid(train_logits).cpu().numpy()
            train_pred = (train_proba >= 0.5).astype(int)
            train_true = data.y[train_mask_t].cpu().numpy()
            train_acc = accuracy_score(train_true, train_pred)

        # Validation metrics
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)[val_mask_t]
            val_proba = torch.sigmoid(val_logits).cpu().numpy()
            val_pred = (val_proba >= 0.5).astype(int)
            val_true = data.y[val_mask_t].cpu().numpy()
            val_acc = accuracy_score(val_true, val_pred)
            val_f1 = f1_score(val_true, val_pred, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

        epoch_list.append(epoch)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        val_f1_hist.append(val_f1)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}: train_loss={loss.item():.4f}, "
                f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_macroF1={val_f1:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save training curve
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epoch_list, train_acc_hist, label="Train Acc")
    ax1.plot(epoch_list, val_acc_hist, label="Val Acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(epoch_list, val_f1_hist, label="Val Macro F1", linestyle="--")
    ax2.set_ylabel("Val Macro F1")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    curve_path = RESULT_DIR / "gnn_training_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] GNN training curve saved to: {curve_path}")

    # Test
    model.eval()
    with torch.no_grad():
        test_logits = model(data.x, data.edge_index)[test_mask_t]
        test_proba = torch.sigmoid(test_logits).cpu().numpy()
        test_pred = (test_proba >= 0.5).astype(int)

    y_test_bin = data.y[test_mask_t].cpu().numpy()
    labels_3_test = labels_3[test_mask]

    return evaluate_binary("GNN", y_test_bin, test_pred, test_proba, labels_3_test)


# ============================================================
# 7. Main
# ============================================================

def main():
    print("Generating synthetic influence graph (binary setting)...")
    data, x_np, y_bin, labels_3 = generate_influence_graph_binary()

    print(f"Graph:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")

    # Feature summary (for slides)
    df_feat = pd.DataFrame(
        x_np,
        columns=["num_followers_feat", "num_following_feat", "num_posts", "num_comments"],
    )
    df_feat["type3"] = labels_3
    print("\n[INFO] Average FEATURES by internal 3-class type (0=normal,1=infl,2=meta):")
    print(df_feat.groupby("type3").mean())

    # Splits
    train_mask, val_mask, test_mask = make_splits(labels_3, 0.6, 0.2, 0.2)

    # Run models
    xgb_metrics = run_xgboost(x_np, y_bin, labels_3, train_mask, val_mask, test_mask)
    mlp_metrics = run_mlp(x_np, y_bin, labels_3, train_mask, val_mask, test_mask)
    gnn_metrics = run_gnn(data, labels_3, train_mask, val_mask, test_mask)

    print("\n\n================= FINAL BINARY COMPARISON =================")
    for m in [xgb_metrics, mlp_metrics, gnn_metrics]:
        print(f"\n{m['name']}:")
        print(f"  Accuracy       : {m['accuracy']:.4f}")
        print(f"  Macro F1       : {m['macro_f1']:.4f}")
        print(f"  Pos F1         : {m['pos_f1']:.4f}")
        print(f"  Meta Precision : {m['meta_precision']:.4f}")
        print(f"  Meta Recall    : {m['meta_recall']:.4f}")
        print(f"  Meta F1        : {m['meta_f1']:.4f}")
        print(f"  AUC            : {m['auc']:.4f}")
        print(f"  CM image       : {m['cm_path']}")
        print(f"  ROC image      : {m['roc_path']}")

    print("\nTraining curves:")
    print(f"  MLP curve      : {RESULT_DIR / 'mlp_training_curve.png'}")
    print(f"  GNN curve      : {RESULT_DIR / 'gnn_training_curve.png'}")

    print("\nAll plots are saved in:", RESULT_DIR)


if __name__ == "__main__":
    main()
