"""
GNN CPU vs GPU Benchmark on Synthetic Influence Graph

Task:
  Binary classification:
    y = 0 -> normal user
    y = 1 -> influencer or meta-influencer

We:
  - Generate the same kind of synthetic follower graph where:
      * normals & meta have similar tabular stats
      * meta are "influencers-of-influencers" structurally.
  - Train multiple GNNs with different hops (layers):
      * Architectures: GraphSAGE, GCN
      * Hops (num_layers): 1, 2, 3, 4
  - For each (arch, hops) and device (CPU/GPU):
      * Measure average training time per epoch
      * Measure inference time (forward pass on all nodes)
      * Report test accuracy & F1 (to confirm correctness)

Outputs:
  - Prints summary table
  - Saves:
      results_gnn_cpu_gpu/cpu_gpu_gnn_timing.csv
      results_gnn_cpu_gpu/gnn_cpu_vs_gpu_training_time.png
      results_gnn_cpu_gpu/gnn_cpu_vs_gpu_inference_time.png
"""

import time
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv

from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# 0. Global config
# ============================================================

RNG_SEED = 42
RESULT_DIR = Path("results_gnn_cpu_gpu")
RESULT_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = RNG_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. Synthetic influence graph generator (binary)
# ============================================================

def generate_influence_graph_binary(
    num_normal: int = 6000,
    num_influencers: int = 3000,
    num_meta: int = 1000,
):
    """
    Build a user-only directed follower graph:
      edge (u -> v) means "u follows v".

    Internal types:
      0 = normal
      1 = influencer
      2 = meta (influencer-of-influencers)

    Binary label:
      y_bin = 0 if type == 0
      y_bin = 1 if type in {1, 2}

    Node features (4D):
      [num_followers_feat, num_following_feat, num_posts, num_comments]
      - normal & meta: similar distributions (to confuse tabular models)
      - influencers: clearly larger stats
    """
    set_seed(RNG_SEED)

    total_nodes = num_normal + num_influencers + num_meta
    node_ids = np.arange(total_nodes)

    normal_ids = node_ids[:num_normal]
    infl_ids = node_ids[num_normal:num_normal + num_influencers]
    meta_ids = node_ids[num_normal + num_influencers:]

    labels_3 = np.zeros(total_nodes, dtype=int)
    labels_3[infl_ids] = 1
    labels_3[meta_ids] = 2

    labels_bin = (labels_3 != 0).astype(int)

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

    # 1) Normals follow mostly influencers + some normals/meta
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

    # 2) Influencers follow metas & other influencers (followers from normals already exist)
    for v in infl_ids:
        k_meta = np.random.poisson(lam=10)
        if len(meta_ids) > 0 and k_meta > 0:
            metas = np.random.choice(meta_ids, size=k_meta, replace=True)
            for m in metas:
                add_edge(v, m)

        k_inf_follow = np.random.poisson(lam=5)
        if len(infl_ids) > 1 and k_inf_follow > 0:
            others = np.random.choice(infl_ids, size=k_inf_follow, replace=True)
            for o in others:
                add_edge(v, o)

    # 3) Meta influencers: followed by many influencers + some normals;
    #    also follow influencers.
    for m in meta_ids:
        k_inf_followers = np.random.poisson(lam=60)
        if len(infl_ids) > 0 and k_inf_followers > 0:
            inf_followers = np.random.choice(infl_ids, size=k_inf_followers, replace=True)
            for u in inf_followers:
                add_edge(u, m)

        k_norm_followers = np.random.poisson(lam=40)
        if len(normal_ids) > 0 and k_norm_followers > 0:
            norm_followers = np.random.choice(normal_ids, size=k_norm_followers, replace=True)
            for u in norm_followers:
                add_edge(u, m)

        k_follow_inf = np.random.poisson(lam=20)
        if len(infl_ids) > 0 and k_follow_inf > 0:
            inf_targets = np.random.choice(infl_ids, size=k_follow_inf, replace=True)
            for v in inf_targets:
                add_edge(m, v)

    src_edges_arr = np.array(src_edges, dtype=int)
    dst_edges_arr = np.array(dst_edges, dtype=int)

    # Tabular features
    num_followers_feat = np.zeros(total_nodes, dtype=float)
    num_following_feat = np.zeros(total_nodes, dtype=float)

    mu_nm_followers = 1200.0
    sigma_nm_followers = 200.0
    mu_nm_following = 600.0
    sigma_nm_following = 100.0

    # normals
    num_followers_feat[normal_ids] = np.random.normal(
        loc=mu_nm_followers, scale=sigma_nm_followers, size=len(normal_ids)
    )
    num_following_feat[normal_ids] = np.random.normal(
        loc=mu_nm_following, scale=sigma_nm_following, size=len(normal_ids)
    )

    # meta (similar to normals)
    num_followers_feat[meta_ids] = np.random.normal(
        loc=mu_nm_followers * 1.02, scale=sigma_nm_followers, size=len(meta_ids)
    )
    num_following_feat[meta_ids] = np.random.normal(
        loc=mu_nm_following * 1.02, scale=sigma_nm_following, size=len(meta_ids)
    )

    # influencers big
    num_followers_feat[infl_ids] = np.random.normal(
        loc=20_000.0, scale=3_000.0, size=len(infl_ids)
    )
    num_following_feat[infl_ids] = np.random.normal(
        loc=2_000.0, scale=400.0, size=len(infl_ids)
    )

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

    return data, labels_3


# ============================================================
# 2. Splits
# ============================================================

def make_splits(labels_3, train_frac=0.6, val_frac=0.2, test_frac=0.2):
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
# 3. GNN architectures
# ============================================================

class StackedGNN(nn.Module):
    """
    Generic stacked GNN:
      - conv_type: "graphsage" or "gcn"
      - num_layers: number of message-passing layers (≈ hops)
    """

    def __init__(self, in_dim, hidden_dim, num_layers, conv_type="graphsage", dropout=0.1):
        super().__init__()
        assert num_layers >= 1
        self.conv_type = conv_type.lower()
        if self.conv_type == "graphsage":
            Conv = SAGEConv
        elif self.conv_type == "gcn":
            Conv = GCNConv
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.layers = nn.ModuleList()
        # first layer
        self.layers.append(Conv(in_dim, hidden_dim))
        # hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(Conv(hidden_dim, hidden_dim))

        self.lin = nn.Linear(hidden_dim, 1)  # binary logit
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        x = self.lin(x).squeeze(-1)
        return x


# ============================================================
# 4. Training + timing helper
# ============================================================

def train_and_time_gnn(
    data: Data,
    train_mask,
    val_mask,
    test_mask,
    arch_name: str,
    num_layers: int,
    hidden_dim: int = 64,
    epochs: int = 40,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """
    Train a GNN on given device and measure:
      - avg training time per epoch
      - inference time (single forward pass on all nodes)
    Also returns test accuracy/F1 for sanity.

    device_str: "cpu" or "cuda"
    """
    device = torch.device(device_str)
    data = data.to(device)
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool, device=device)
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool, device=device)

    num_features = data.x.size(1)
    model = StackedGNN(
        in_dim=num_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        conv_type=arch_name,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    criterion = nn.BCEWithLogitsLoss()

    epoch_times = []
    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        start_t = time.time()

        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask_t], data.y[train_mask_t].float())
        loss.backward()
        optimizer.step()

        # Record end of epoch
        end_t = time.time()
        epoch_times.append(end_t - start_t)

        # Quick val F1 for "correctness" tracking
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)[val_mask_t]
            val_proba = torch.sigmoid(val_logits).cpu().numpy()
            val_pred = (val_proba >= 0.5).astype(int)
            val_true = data.y[val_mask_t].cpu().numpy()
            val_f1 = f1_score(val_true, val_pred, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

        if epoch % 10 == 0:
            # Also compute train/val accuracy for logging
            with torch.no_grad():
                train_logits = logits[train_mask_t]
                train_proba = torch.sigmoid(train_logits).cpu().numpy()
                train_pred = (train_proba >= 0.5).astype(int)
                train_true = data.y[train_mask_t].cpu().numpy()
                train_acc = accuracy_score(train_true, train_pred)

                val_acc = accuracy_score(val_true, val_pred)

            print(
                f"[{device_str.upper()}] {arch_name} (layers={num_layers}) "
                f"Epoch {epoch:03d}: loss={loss.item():.4f}, "
                f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_F1={val_f1:.4f}"
            )

    avg_train_time = float(np.mean(epoch_times))

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Measure inference time (forward pass on all nodes), averaged
    model.eval()
    N_runs = 10
    inf_times = []
    with torch.no_grad():
        for _ in range(N_runs):
            start_t = time.time()
            _ = model(data.x, data.edge_index)
            if device_str == "cuda":
                torch.cuda.synchronize()
            end_t = time.time()
            inf_times.append(end_t - start_t)
    avg_inf_time = float(np.mean(inf_times))

    # Final test metrics
    with torch.no_grad():
        test_logits = model(data.x, data.edge_index)[test_mask_t]
        test_proba = torch.sigmoid(test_logits).cpu().numpy()
        test_pred = (test_proba >= 0.5).astype(int)
        test_true = data.y[test_mask_t].cpu().numpy()

    test_acc = accuracy_score(test_true, test_pred)
    test_f1 = f1_score(test_true, test_pred, average="macro")

    return {
        "arch": arch_name,
        "layers": num_layers,
        "device": device_str,
        "avg_train_time_s": avg_train_time,
        "avg_inference_time_s": avg_inf_time,
        "test_acc": test_acc,
        "test_f1": test_f1,
    }


# ============================================================
# 5. Main benchmark
# ============================================================

def main():
    print("Generating synthetic influence graph for GNN CPU vs GPU benchmark...")
    data, labels_3 = generate_influence_graph_binary()
    print(f"Graph: nodes={data.num_nodes}, edges={data.num_edges}")

    train_mask, val_mask, test_mask = make_splits(labels_3, 0.6, 0.2, 0.2)

    print("\n[INFO] Class counts (3-type) in full graph:")
    unique, counts = np.unique(labels_3, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  type3={cls}: {cnt}")

    # Check GPU
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("\nCUDA available. Will benchmark both CPU and GPU.")
    else:
        print("\nCUDA NOT available. Will only run CPU benchmarks.")

    # Config: architectures and hops (= layers)
    architectures = ["graphsage", "gcn"]
    hop_list = [1, 2, 3, 4]
    epochs = 40

    results = []

    for arch in architectures:
        for hops in hop_list:
            print(f"\n=== Benchmark: {arch}, hops={hops} ===")
            # CPU
            cpu_res = train_and_time_gnn(
                data=data.clone(),
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                arch_name=arch,
                num_layers=hops,
                hidden_dim=64,
                epochs=epochs,
                device_str="cpu",
            )
            results.append(cpu_res)

            # GPU (if available)
            if has_cuda:
                gpu_res = train_and_time_gnn(
                    data=data.clone(),
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                    arch_name=arch,
                    num_layers=hops,
                    hidden_dim=64,
                    epochs=epochs,
                    device_str="cuda",
                )
                results.append(gpu_res)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    csv_path = RESULT_DIR / "cpu_gpu_gnn_timing.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Timing results saved to: {csv_path}")
    print(df)

    # Plot training time comparison
    if has_cuda:
        fig_train, ax_train = plt.subplots(figsize=(8, 5))
        for arch in architectures:
            for hops in hop_list:
                subset_cpu = df[
                    (df["arch"] == arch)
                    & (df["layers"] == hops)
                    & (df["device"] == "cpu")
                ]
                subset_gpu = df[
                    (df["arch"] == arch)
                    & (df["layers"] == hops)
                    & (df["device"] == "cuda")
                ]
                if len(subset_cpu) == 0 or len(subset_gpu) == 0:
                    continue
                label = f"{arch}-hops{hops}"
                cpu_t = subset_cpu["avg_train_time_s"].values[0]
                gpu_t = subset_gpu["avg_train_time_s"].values[0]
                ax_train.bar(
                    label + "-CPU",
                    cpu_t,
                )
                ax_train.bar(
                    label + "-GPU",
                    gpu_t,
                )

        ax_train.set_ylabel("Avg training time per epoch (s)")
        ax_train.set_title("GNN CPU vs GPU – Training Time")
        plt.xticks(rotation=45, ha="right")
        fig_train.tight_layout()
        train_plot_path = RESULT_DIR / "gnn_cpu_vs_gpu_training_time.png"
        fig_train.savefig(train_plot_path, dpi=150)
        plt.close(fig_train)
        print(f"[INFO] Training time plot saved to: {train_plot_path}")

        # Plot inference time comparison
        fig_inf, ax_inf = plt.subplots(figsize=(8, 5))
        for arch in architectures:
            for hops in hop_list:
                subset_cpu = df[
                    (df["arch"] == arch)
                    & (df["layers"] == hops)
                    & (df["device"] == "cpu")
                ]
                subset_gpu = df[
                    (df["arch"] == arch)
                    & (df["layers"] == hops)
                    & (df["device"] == "cuda")
                ]
                if len(subset_cpu) == 0 or len(subset_gpu) == 0:
                    continue
                label = f"{arch}-hops{hops}"
                cpu_t = subset_cpu["avg_inference_time_s"].values[0]
                gpu_t = subset_gpu["avg_inference_time_s"].values[0]
                ax_inf.bar(
                    label + "-CPU",
                    cpu_t,
                )
                ax_inf.bar(
                    label + "-GPU",
                    gpu_t,
                )

        ax_inf.set_ylabel("Avg inference time (s) for full graph")
        ax_inf.set_title("GNN CPU vs GPU – Inference Time")
        plt.xticks(rotation=45, ha="right")
        fig_inf.tight_layout()
        inf_plot_path = RESULT_DIR / "gnn_cpu_vs_gpu_inference_time.png"
        fig_inf.savefig(inf_plot_path, dpi=150)
        plt.close(fig_inf)
        print(f"[INFO] Inference time plot saved to: {inf_plot_path}")
    else:
        print("\n[INFO] Skipping GPU comparison plots since CUDA is not available.")


if __name__ == "__main__":
    main()
