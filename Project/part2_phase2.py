# ==========================================================
# PHASE 2 - PART 2 (CPU vs GPU) - GNN ONLY + HOPS SWEEP
# StackOverflow + WikiVote
# NO PyG / NO torch-scatter needed
# Uses Sparse adjacency with torch.sparse.mm
# ==========================================================

import os, time, json, random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

# =========================
# Drive mount (Colab)
# =========================
from google.colab import drive
drive.mount("/content/drive")


# ==========================================================
# Config
# ==========================================================

@dataclass
class CFG:
    # meta labeling config (same as your Part-1)
    meta_frac: float = 0.20
    meta_rep_max_quantile: float = 0.70
    meta_min_influ_receivers: int = 1

    # StackOverflow API config
    so_site: str = "stackoverflow"
    so_tag: str = "machine-learning"
    so_pagesize: int = 100
    so_max_pages_questions: int = 3
    so_max_pages_answers: int = 3
    so_sleep_s: float = 0.35
    so_api_key: Optional[str] = None

    # influencer label
    influencer_top_pct: float = 0.10

    # splits
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # GNN training
    lr: float = 3e-4
    hidden: int = 64
    dropout: float = 0.2
    epochs: int = 80

    # Part-2 sweep
    hops_list: Tuple[int, ...] = (1, 2, 3, 4)
    gnn_list: Tuple[str, ...] = ("sage", "gcn", "sgc")  # different GNNs

    # timing
    infer_runs: int = 20

    # device
    use_cuda_if_available: bool = True

    # output/cache in Drive
    out_dir: str = "/content/drive/MyDrive/results_phase2_part2_cpu_gpu"
    cache_dir: str = "/content/drive/MyDrive/cache_phase2_real"


cfg = CFG()


# ==========================================================
# Utils (from your Part-1)
# ==========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def split_masks(n, train_frac, val_frac, test_frac, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    m_train = np.zeros(n, dtype=bool)
    m_val = np.zeros(n, dtype=bool)
    m_test = np.zeros(n, dtype=bool)
    m_train[train_idx] = True
    m_val[val_idx] = True
    m_test[test_idx] = True
    return m_train, m_val, m_test

def standardize_from_train(X_full: np.ndarray, train_mask: np.ndarray):
    Xtr = X_full[train_mask]
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xn = (X_full - mu) / sd
    return Xn.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)

def compute_pos_weight(y_train: np.ndarray) -> float:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    return float(neg / max(1, pos))

def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def plot_time_curves(df: pd.DataFrame, dataset_name: str):
    # train epoch time
    plt.figure()
    for gnn in cfg.gnn_list:
        for dev in ["cpu", "cuda"]:
            d = df[(df["gnn"] == gnn) & (df["device"] == dev)]
            if len(d) == 0:
                continue
            plt.plot(d["hops"], d["avg_epoch_time_s"], label=f"{gnn}-{dev}")
    plt.xlabel("Hops (layers)")
    plt.ylabel("Avg epoch time (s)")
    plt.title(f"{dataset_name}: Train time vs hops (CPU vs GPU)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(cfg.out_dir) / f"{dataset_name}_train_time_vs_hops.png"), dpi=160)
    plt.close()

    # inference time
    plt.figure()
    for gnn in cfg.gnn_list:
        for dev in ["cpu", "cuda"]:
            d = df[(df["gnn"] == gnn) & (df["device"] == dev)]
            if len(d) == 0:
                continue
            plt.plot(d["hops"], d["avg_infer_time_s"], label=f"{gnn}-{dev}")
    plt.xlabel("Hops (layers)")
    plt.ylabel("Avg forward time (s)")
    plt.title(f"{dataset_name}: Inference time vs hops (CPU vs GPU)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(cfg.out_dir) / f"{dataset_name}_infer_time_vs_hops.png"), dpi=160)
    plt.close()


# ==========================================================
# Sparse adjacency builders
# ==========================================================

def build_row_normalized_adj(n: int, edges_u: np.ndarray, edges_v: np.ndarray, device: torch.device):
    # add self-loops
    self_u = np.arange(n, dtype=np.int64)
    self_v = np.arange(n, dtype=np.int64)
    u = np.concatenate([edges_u, self_u])
    v = np.concatenate([edges_v, self_v])

    # row-normalized: D^{-1} A
    deg = np.bincount(u, minlength=n).astype(np.float32)
    deg = np.where(deg < 1.0, 1.0, deg)
    vals = (1.0 / deg[u]).astype(np.float32)

    idx = torch.tensor(np.vstack([u, v]), dtype=torch.long, device=device)
    val = torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce()

def build_gcn_norm_adj(n: int, edges_u: np.ndarray, edges_v: np.ndarray, device: torch.device):
    # symmetric normalized: D^{-1/2} (A+I) D^{-1/2}
    self_u = np.arange(n, dtype=np.int64)
    self_v = np.arange(n, dtype=np.int64)
    u = np.concatenate([edges_u, self_u])
    v = np.concatenate([edges_v, self_v])

    deg = np.bincount(u, minlength=n).astype(np.float32)
    deg = np.where(deg < 1.0, 1.0, deg)

    inv_sqrt = 1.0 / np.sqrt(deg)
    vals = (inv_sqrt[u] * inv_sqrt[v]).astype(np.float32)

    idx = torch.tensor(np.vstack([u, v]), dtype=torch.long, device=device)
    val = torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce()


# ==========================================================
# GNN Variants (NO PyG)
# ==========================================================

class SparseGraphSAGE(nn.Module):
    # mean aggregation via row-normalized adj
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.self_lins = nn.ModuleList()
        self.nei_lins  = nn.ModuleList()
        dims = [in_dim] + [hidden] * layers
        for k in range(layers):
            self.self_lins.append(nn.Linear(dims[k], hidden))
            self.nei_lins.append(nn.Linear(dims[k], hidden))
        self.out_lin = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adj_row_norm: torch.Tensor):
        h = x
        for k in range(self.layers):
            nei = torch.sparse.mm(adj_row_norm, h)
            h = self.self_lins[k](h) + self.nei_lins[k](nei)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out_lin(h).squeeze(-1)


class SparseGCN(nn.Module):
    # classic GCN using symmetric normalized adjacency
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.lins = nn.ModuleList()
        dims = [in_dim] + [hidden] * layers
        for k in range(layers):
            self.lins.append(nn.Linear(dims[k], hidden))
        self.out_lin = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adj_gcn: torch.Tensor):
        h = x
        for k in range(self.layers):
            h = torch.sparse.mm(adj_gcn, h)
            h = self.lins[k](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out_lin(h).squeeze(-1)


class SGC(nn.Module):
    # Simplified Graph Convolution: (A_norm^K X) -> Linear
    def __init__(self, in_dim: int, hidden: int, hops: int, dropout: float):
        super().__init__()
        self.hops = hops
        self.dropout = dropout
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adj_gcn: torch.Tensor):
        h = x
        # propagate hops times (no nonlinearity between)
        for _ in range(self.hops):
            h = torch.sparse.mm(adj_gcn, h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h).squeeze(-1)


# ==========================================================
# Core Train/Eval + Timing
# ==========================================================

@torch.no_grad()
def timed_inference(model, X, adj, device, runs=20):
    model.eval()
    # warmup
    _ = model(X, adj)
    sync_if_cuda(device)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(X, adj)
        sync_if_cuda(device)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))

def train_eval_one(model, X, y, tr_mask, va_mask, te_mask, adj, device, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    pw = compute_pos_weight(y[tr_mask].detach().cpu().numpy().astype(int))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    best_f1 = -1.0
    best_state = None

    epoch_times = []

    # warmup step for stable GPU timing
    model.train()
    opt.zero_grad()
    logits = model(X, adj)
    loss = crit(logits[tr_mask], y[tr_mask])
    loss.backward()
    opt.step()
    sync_if_cuda(device)

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()

        model.train()
        opt.zero_grad()
        logits = model(X, adj)
        loss = crit(logits[tr_mask], y[tr_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sync_if_cuda(device)

        epoch_times.append(time.perf_counter() - t0)

        # val f1
        model.eval()
        with torch.no_grad():
            v_logits = model(X, adj)[va_mask]
            v_prob = torch.sigmoid(v_logits).detach().cpu().numpy()
            v_pred = (v_prob >= 0.5).astype(int)
            v_true = y[va_mask].detach().cpu().numpy().astype(int)
            v_f1 = f1_score(v_true, v_pred, average="macro", zero_division=0)

        if v_f1 > best_f1:
            best_f1 = v_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # test metrics
    model.eval()
    with torch.no_grad():
        t_logits = model(X, adj)[te_mask]
        t_prob = torch.sigmoid(t_logits).detach().cpu().numpy()
        t_pred = (t_prob >= 0.5).astype(int)
        t_true = y[te_mask].detach().cpu().numpy().astype(int)

    acc = accuracy_score(t_true, t_pred)
    f1m = f1_score(t_true, t_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(t_true, t_prob) if len(np.unique(t_true)) == 2 else float("nan")
    except Exception:
        auc = float("nan")

    avg_epoch = float(np.mean(epoch_times)) if len(epoch_times) else float("nan")
    return acc, f1m, auc, avg_epoch


def run_cpu_gpu_benchmark(dataset_name: str, user_df: pd.DataFrame, graph: Dict, meta: Dict):
    feature_cols = meta["feature_cols_user"]
    X_full = user_df[feature_cols].values.astype(np.float32)
    y_full = user_df["y"].values.astype(int)

    n = len(user_df)
    m_train, m_val, m_test = split_masks(n, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)
    X_full, mu, sd = standardize_from_train(X_full, m_train)

    # masks
    tr = torch.tensor(m_train, dtype=torch.bool)
    va = torch.tensor(m_val, dtype=torch.bool)
    te = torch.tensor(m_test, dtype=torch.bool)

    # edges
    edges_u = graph["edges_u"].astype(np.int64)
    edges_v = graph["edges_v"].astype(np.int64)

    # devices to test
    devices = [torch.device("cpu")]
    if cfg.use_cuda_if_available and torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    rows = []

    for dev in devices:
        # move data
        X = torch.tensor(X_full, dtype=torch.float32, device=dev)
        y = torch.tensor(y_full.astype(np.float32), dtype=torch.float32, device=dev)

        # build both adj types once per device
        adj_row = build_row_normalized_adj(n, edges_u, edges_v, dev)
        adj_gcn = build_gcn_norm_adj(n, edges_u, edges_v, dev)

        tr_d, va_d, te_d = tr.to(dev), va.to(dev), te.to(dev)

        for gnn_name in cfg.gnn_list:
            for hops in cfg.hops_list:
                print(f"\n[{dataset_name}] gnn={gnn_name} hops={hops} device={dev.type}")

                if gnn_name == "sage":
                    model = SparseGraphSAGE(in_dim=X.shape[1], hidden=cfg.hidden, layers=hops, dropout=cfg.dropout).to(dev)
                    adj = adj_row
                elif gnn_name == "gcn":
                    model = SparseGCN(in_dim=X.shape[1], hidden=cfg.hidden, layers=hops, dropout=cfg.dropout).to(dev)
                    adj = adj_gcn
                elif gnn_name == "sgc":
                    model = SGC(in_dim=X.shape[1], hidden=cfg.hidden, hops=hops, dropout=cfg.dropout).to(dev)
                    adj = adj_gcn
                else:
                    raise ValueError("Unknown gnn_name")

                # train + eval
                acc, f1m, auc, avg_epoch_time = train_eval_one(
                    model, X, y, tr_d, va_d, te_d, adj, dev,
                    epochs=cfg.epochs, lr=cfg.lr
                )

                # inference timing
                avg_infer_time = timed_inference(model, X, adj, dev, runs=cfg.infer_runs)

                row = dict(
                    dataset=dataset_name,
                    gnn=gnn_name,
                    hops=int(hops),
                    device=dev.type,
                    test_acc=float(acc),
                    test_macro_f1=float(f1m),
                    test_auc=float(auc),
                    avg_epoch_time_s=float(avg_epoch_time),
                    avg_infer_time_s=float(avg_infer_time),
                    n_users=int(meta["n_users"]),
                    n_edges=int(meta["n_edges"]),
                )
                rows.append(row)
                print(row)

    df = pd.DataFrame(rows)

    # save
    out_csv = Path(cfg.out_dir) / f"{dataset_name}_part2_cpu_gpu_gnn_only.csv"
    df.to_csv(out_csv, index=False)

    save_json(meta, str(Path(cfg.out_dir) / f"{dataset_name}_meta.json"))

    # plots
    plot_time_curves(df, dataset_name)

    print("\nSaved:", out_csv)
    return df


# ==========================================================
# YOUR PART-1 BUILDERS (copied exactly from your working code)
# ==========================================================

def so_api_get(endpoint: str, params: Dict) -> Dict:
    if not HAS_REQUESTS:
        raise RuntimeError("requests not installed. Run: pip install requests")

    base = "https://api.stackexchange.com/2.3"
    url = base + endpoint

    params = dict(params)
    params["site"] = cfg.so_site
    params["pagesize"] = cfg.so_pagesize
    if cfg.so_api_key:
        params["key"] = cfg.so_api_key

    r = requests.get(url, params=params, headers={"User-Agent": "phase2-realdata/1.0"})
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")
    data = r.json()

    if "backoff" in data:
        time.sleep(float(data["backoff"]) + cfg.so_sleep_s)
    time.sleep(cfg.so_sleep_s)
    return data

def so_fetch_questions_by_tag() -> List[Dict]:
    cache_path = Path(cfg.cache_dir) / f"so_questions_{cfg.so_site}_{cfg.so_tag}.json"
    if cache_path.exists():
        return load_json(cache_path)

    all_items = []
    for page in range(1, cfg.so_max_pages_questions + 1):
        data = so_api_get(
            "/questions",
            params={"page": page, "order": "desc", "sort": "votes", "tagged": cfg.so_tag},
        )
        all_items.extend(data.get("items", []))
        if not data.get("has_more", False):
            break

    save_json(all_items, cache_path)
    return all_items

def so_fetch_answers_for_questions(question_ids: List[int]) -> List[Dict]:
    cache_path = Path(cfg.cache_dir) / f"so_answers_{cfg.so_site}_{cfg.so_tag}.json"
    if cache_path.exists():
        return load_json(cache_path)

    all_items = []
    chunk = 100
    for i in range(0, len(question_ids), chunk):
        ids = ";".join(map(str, question_ids[i:i+chunk]))
        for page in range(1, cfg.so_max_pages_answers + 1):
            data = so_api_get(
                f"/questions/{ids}/answers",
                params={"page": page, "order": "desc", "sort": "votes"},
            )
            all_items.extend(data.get("items", []))
            if not data.get("has_more", False):
                break

    save_json(all_items, cache_path)
    return all_items

def build_stackoverflow_dataset() -> Tuple[pd.DataFrame, Dict, Dict]:
    questions = so_fetch_questions_by_tag()
    q_ids = [q["question_id"] for q in questions]
    answers = so_fetch_answers_for_questions(q_ids)

    def owner_user_id(obj):
        return obj.get("owner", {}).get("user_id", None)

    user_ids = set()
    for q in questions:
        uid = owner_user_id(q)
        if uid is not None:
            user_ids.add(uid)
    for a in answers:
        uid = owner_user_id(a)
        if uid is not None:
            user_ids.add(uid)

    user_ids = sorted(list(user_ids))
    user2idx = {u: i for i, u in enumerate(user_ids)}

    u_stats = {u: {
        "num_questions": 0,
        "num_answers": 0,
        "sum_q_score": 0.0,
        "sum_a_score": 0.0,
        "distinct_users_interacted": set(),
        "distinct_posts_interacted": set(),
    } for u in user_ids}

    q_owner = {q["question_id"]: owner_user_id(q) for q in questions}

    for q in questions:
        uid = owner_user_id(q)
        if uid is None or uid not in u_stats:
            continue
        u_stats[uid]["num_questions"] += 1
        u_stats[uid]["sum_q_score"] += float(q.get("score", 0))
        u_stats[uid]["distinct_posts_interacted"].add(int(q["question_id"]))

    for a in answers:
        uid = owner_user_id(a)
        if uid is None or uid not in u_stats:
            continue
        u_stats[uid]["num_answers"] += 1
        u_stats[uid]["sum_a_score"] += float(a.get("score", 0))
        qid = a.get("question_id", None)
        if qid is not None:
            u_stats[uid]["distinct_posts_interacted"].add(int(qid))

        target = q_owner.get(qid, None)
        if target is not None and target in u_stats and target != uid:
            u_stats[uid]["distinct_users_interacted"].add(target)

    rows = []
    for u in user_ids:
        s = u_stats[u]
        rep_proxy = (
            4.0 * s["sum_q_score"] +
            6.0 * s["sum_a_score"] +
            2.0 * s["num_answers"] +
            1.5 * s["num_questions"]
        )
        rows.append({
            "user_id": u,
            "num_questions": s["num_questions"],
            "num_answers": s["num_answers"],
            "sum_q_score": s["sum_q_score"],
            "sum_a_score": s["sum_a_score"],
            "rep_proxy": rep_proxy,
            "distinct_users_interacted": len(s["distinct_users_interacted"]),
            "distinct_posts_interacted": len(s["distinct_posts_interacted"]),
        })

    user_df = pd.DataFrame(rows)

    thr = np.quantile(user_df["rep_proxy"].values, 1.0 - cfg.influencer_top_pct)
    user_df["is_influencer"] = (user_df["rep_proxy"] >= thr).astype(int)
    influencer_set = set(user_df.loc[user_df["is_influencer"] == 1, "user_id"].tolist())

    influ_recv = {u: set() for u in user_ids}
    for a in answers:
        influ = owner_user_id(a)
        if influ is None or influ not in influencer_set:
            continue
        qid = a.get("question_id", None)
        target = q_owner.get(qid, None)
        if target is None or target not in influ_recv:
            continue
        if target != influ:
            influ_recv[target].add(influ)

    user_df["influ_interactions_received"] = user_df["user_id"].apply(lambda uid: len(influ_recv.get(uid, set()))).astype(int)

    rep_cap = np.quantile(user_df["rep_proxy"].values, cfg.meta_rep_max_quantile)
    cand = user_df[
        (user_df["is_influencer"] == 0) &
        (user_df["rep_proxy"] <= rep_cap) &
        (user_df["influ_interactions_received"] >= cfg.meta_min_influ_receivers)
    ].copy()

    K = max(1, int(cfg.meta_frac * len(user_df)))
    cand = cand.sort_values("influ_interactions_received", ascending=False)
    meta_ids = set(cand["user_id"].head(K).tolist())

    user_df["is_meta"] = user_df["user_id"].isin(meta_ids).astype(int)
    user_df["y"] = ((user_df["is_influencer"] == 1) | (user_df["is_meta"] == 1)).astype(int)

    edges_u, edges_v = [], []
    for u in user_ids:
        uidx = user2idx[u]
        for v in u_stats[u]["distinct_users_interacted"]:
            if v in user2idx:
                edges_u.append(uidx)
                edges_v.append(user2idx[v])
    edges_u = np.array(edges_u, dtype=np.int64)
    edges_v = np.array(edges_v, dtype=np.int64)

    feature_cols = [
        "num_questions", "num_answers", "sum_q_score", "sum_a_score",
        "distinct_users_interacted", "distinct_posts_interacted", "rep_proxy"
    ]

    meta = {
        "dataset": "stackoverflow",
        "tag": cfg.so_tag,
        "n_users": int(len(user_df)),
        "n_edges": int(len(edges_u)),
        "thr_rep_proxy": float(thr),
        "label_stats": {
            "positives": int(user_df["y"].sum()),
            "negatives": int((user_df["y"] == 0).sum()),
            "influencers": int(user_df["is_influencer"].sum()),
            "metas": int(user_df["is_meta"].sum()),
        },
        "feature_cols_user": feature_cols,
    }

    graph = {"edges_u": edges_u, "edges_v": edges_v, "feature_cols": feature_cols}
    return user_df, graph, meta


def download_wikivote(out_path: str):
    import urllib.request
    url = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"
    if not os.path.exists(out_path):
        print(f"Downloading {url} -> {out_path}")
        urllib.request.urlretrieve(url, out_path)

def load_wikivote_edges(cache_dir: str) -> Tuple[np.ndarray, np.ndarray, int]:
    import gzip
    gz_path = os.path.join(cache_dir, "wiki-Vote.txt.gz")
    download_wikivote(gz_path)

    edges = []
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            edges.append((int(parts[0]), int(parts[1])))

    nodes = sorted(set([u for u, _ in edges] + [v for _, v in edges]))
    node2idx = {nid: i for i, nid in enumerate(nodes)}
    u = np.array([node2idx[a] for a, _ in edges], dtype=np.int64)
    v = np.array([node2idx[b] for _, b in edges], dtype=np.int64)
    n = len(nodes)
    return u, v, n

def build_wikivote_dataset() -> Tuple[pd.DataFrame, Dict, Dict]:
    u, v, n = load_wikivote_edges(cfg.cache_dir)

    out_deg = np.bincount(u, minlength=n).astype(np.float32)
    in_deg = np.bincount(v, minlength=n).astype(np.float32)
    rep_proxy = 2.0 * in_deg + 1.0 * out_deg

    user_df = pd.DataFrame({
        "user_id": np.arange(n, dtype=np.int64),
        "in_deg": in_deg,
        "out_deg": out_deg,
        "rep_proxy": rep_proxy,
    })

    thr = np.quantile(user_df["rep_proxy"].values, 1.0 - cfg.influencer_top_pct)
    user_df["is_influencer"] = (user_df["rep_proxy"] >= thr).astype(int)
    influencer_idx = set(user_df.index[user_df["is_influencer"] == 1].tolist())

    recv_sets = [set() for _ in range(n)]
    for uu, vv in zip(u.tolist(), v.tolist()):
        if uu in influencer_idx:
            recv_sets[vv].add(uu)
    user_df["influ_interactions_received"] = np.array([len(s) for s in recv_sets], dtype=np.int32)

    rep_cap = np.quantile(user_df["rep_proxy"].values, cfg.meta_rep_max_quantile)
    cand = user_df[
        (user_df["is_influencer"] == 0) &
        (user_df["rep_proxy"] <= rep_cap) &
        (user_df["influ_interactions_received"] >= cfg.meta_min_influ_receivers)
    ].copy()

    K = max(1, int(cfg.meta_frac * len(user_df)))
    cand = cand.sort_values("influ_interactions_received", ascending=False)
    meta_ids = set(cand["user_id"].head(K).tolist())

    user_df["is_meta"] = user_df["user_id"].isin(meta_ids).astype(int)
    user_df["y"] = ((user_df["is_influencer"] == 1) | (user_df["is_meta"] == 1)).astype(int)

    feature_cols = ["in_deg", "out_deg", "rep_proxy"]
    meta = {
        "dataset": "wikivote",
        "n_users": int(n),
        "n_edges": int(len(u)),
        "thr_rep_proxy": float(thr),
        "label_stats": {
            "positives": int(user_df["y"].sum()),
            "negatives": int((user_df["y"] == 0).sum()),
            "influencers": int(user_df["is_influencer"].sum()),
            "metas": int(user_df["is_meta"].sum()),
        },
        "feature_cols_user": feature_cols,
    }
    graph = {"edges_u": u, "edges_v": v, "feature_cols": feature_cols}
    return user_df, graph, meta


# ==========================================================
# MAIN (runs both datasets)
# ==========================================================

def main():
    set_seed(cfg.seed)
    ensure_dirs()

    # save config
    save_json(cfg.__dict__, str(Path(cfg.out_dir) / "part2_cfg.json"))

    print("Output dir:", cfg.out_dir)
    print("Cache dir :", cfg.cache_dir)
    print("CUDA available:", torch.cuda.is_available())

    # ---------- StackOverflow ----------
    print("\n==============================")
    print("PHASE 2 - PART 2A: STACKOVERFLOW (CPU vs GPU)")
    print("==============================")
    user_df_so, graph_so, meta_so = build_stackoverflow_dataset()
    print(json.dumps(meta_so, indent=2))
    df_so = run_cpu_gpu_benchmark("stackoverflow_ml", user_df_so, graph_so, meta_so)

    # ---------- WikiVote ----------
    print("\n==============================")
    print("PHASE 2 - PART 2B: WIKIVOTE (CPU vs GPU)")
    print("==============================")
    user_df_wv, graph_wv, meta_wv = build_wikivote_dataset()
    print(json.dumps(meta_wv, indent=2))
    df_wv = run_cpu_gpu_benchmark("wikivote", user_df_wv, graph_wv, meta_wv)

    # combined
    df_all = pd.concat([df_so, df_wv], axis=0, ignore_index=True)
    out_all = Path(cfg.out_dir) / "ALL_part2_cpu_gpu_gnn_only.csv"
    df_all.to_csv(out_all, index=False)
    print("\n✅ Done. Saved combined CSV:", out_all)


if __name__ == "__main__":
    main()
