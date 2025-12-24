# phase2_realdata_colab_fixed_v2.py
# FIXED: meta_mask alignment bug (pass test-only mask into evaluate_and_save)
# StackOverflow + WikiVote, no Reddit.

import os
import time
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False


# ==========================================================
# Config
# ==========================================================

@dataclass
class CFG:
    meta_frac: float = 0.20
    meta_rep_max_quantile: float = 0.70
    meta_min_influ_receivers: int = 1

    so_site: str = "stackoverflow"
    so_tag: str = "machine-learning"
    so_pagesize: int = 100
    so_max_pages_questions: int = 3
    so_max_pages_answers: int = 3

    so_sleep_s: float = 0.35
    so_api_key: Optional[str] = None

    influencer_top_pct: float = 0.10

    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    mlp_epochs: int = 80
    gnn_epochs: int = 120
    lr: float = 3e-4
    hidden: int = 64
    dropout: float = 0.2
    gnn_hops: int = 3

    use_cuda_if_available: bool = True

    out_dir: str = "/content/drive/MyDrive/results_phase2_real"
    cache_dir: str = "/content/drive/MyDrive/cache_phase2_real"


cfg = CFG()


# ==========================================================
# Utilities
# ==========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

def get_device():
    if cfg.use_cuda_if_available and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_confusion(cm, labels, title, save_path):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=20)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_roc(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

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

def plot_training_curves(hist: Dict, title: str, save_path: str):
    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_acc"], label="val_acc")
    plt.plot(hist["epoch"], hist["val_f1"], label="val_f1")
    plt.xlabel("Epoch")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def evaluate_and_save(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray,
                      out_prefix: str, out_dir: str, meta_mask_test: Optional[np.ndarray] = None):
    """
    IMPORTANT:
      meta_mask_test MUST be aligned with y_true (i.e., test-only length).
    """
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")
    cm = confusion_matrix(y_true, y_pred)

    plot_confusion(cm, labels=["normal(0)", "influence(1)"],
                   title=f"{out_prefix}: {name} Confusion Matrix",
                   save_path=str(Path(out_dir) / f"{out_prefix}_{name}_cm.png"))
    plot_roc(y_true, y_score,
             title=f"{out_prefix}: {name} ROC",
             save_path=str(Path(out_dir) / f"{out_prefix}_{name}_roc.png"))

    rep = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    res = {
        "model": name,
        "accuracy": float(acc),
        "macro_f1": float(f1m),
        "auc": float(auc),
        "precision_pos": float(rep["1"]["precision"]) if "1" in rep else None,
        "recall_pos": float(rep["1"]["recall"]) if "1" in rep else None,
        "f1_pos": float(rep["1"]["f1-score"]) if "1" in rep else None,
    }

    if meta_mask_test is not None:
        meta_mask_test = meta_mask_test.astype(bool)
        meta_count = int(meta_mask_test.sum())
        res["meta_count"] = meta_count
        if meta_count > 0:
            yt = y_true[meta_mask_test]
            yp = y_pred[meta_mask_test]
            prec = float(((yp == 1) & (yt == 1)).sum() / max(1, (yp == 1).sum()))
            rec = float(((yp == 1) & (yt == 1)).sum() / max(1, (yt == 1).sum()))
            f1 = float(2 * prec * rec / max(1e-9, prec + rec))
            res["meta_prec"] = prec
            res["meta_rec"] = rec
            res["meta_f1"] = f1
        else:
            res["meta_prec"] = 0.0
            res["meta_rec"] = 0.0
            res["meta_f1"] = 0.0

    return res


# ==========================================================
# Models
# ==========================================================

class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def run_xgboost(train_X, train_y, val_X, val_y, test_X, test_y):
    if not HAS_XGB:
        return None, None
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
    prob = model.predict_proba(test_X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob

def run_mlp(train_X, train_y, val_X, val_y, X_full, out_prefix):
    dev = get_device()
    Xtr = torch.tensor(train_X, dtype=torch.float32, device=dev)
    ytr = torch.tensor(train_y, dtype=torch.float32, device=dev)
    Xva = torch.tensor(val_X, dtype=torch.float32, device=dev)
    Xall = torch.tensor(X_full, dtype=torch.float32, device=dev)

    model = TabularMLP(in_dim=train_X.shape[1], hidden=cfg.hidden, dropout=cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-5)

    pw = compute_pos_weight(train_y)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=dev))

    hist = {"epoch": [], "train_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1
    best_state = None

    for epoch in range(1, cfg.mlp_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = crit(logits, ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(Xva)
            v_prob = torch.sigmoid(v_logits).detach().cpu().numpy()
            v_pred = (v_prob >= 0.5).astype(int)
            v_acc = accuracy_score(val_y, v_pred)
            v_f1  = f1_score(val_y, v_pred, average="macro", zero_division=0)

        hist["epoch"].append(epoch)
        hist["train_loss"].append(float(loss.item()))
        hist["val_acc"].append(float(v_acc))
        hist["val_f1"].append(float(v_f1))

        if v_f1 > best_f1:
            best_f1 = v_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"[MLP] Epoch {epoch:03d}: loss={loss.item():.4f}, val_acc={v_acc:.4f}, val_f1={v_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        prob_full = torch.sigmoid(model(Xall)).detach().cpu().numpy()
        pred_full = (prob_full >= 0.5).astype(int)

    plot_training_curves(
        hist,
        title=f"{out_prefix}: MLP Training Curves",
        save_path=str(Path(cfg.out_dir) / f"{out_prefix}_mlp_training_curves.png"),
    )

    return pred_full, prob_full


class SparseGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.self_lins = nn.ModuleList()
        self.nei_lins = nn.ModuleList()
        dims = [in_dim] + [hidden] * layers
        for k in range(layers):
            self.self_lins.append(nn.Linear(dims[k], hidden))
            self.nei_lins.append(nn.Linear(dims[k], hidden))
        self.out_lin = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h = x
        for k in range(self.layers):
            nei = torch.sparse.mm(adj, h)
            h = self.self_lins[k](h) + self.nei_lins[k](nei)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out_lin(h).squeeze(-1)

def build_sparse_adj(n: int, edges_u: np.ndarray, edges_v: np.ndarray, device: torch.device):
    self_u = np.arange(n, dtype=np.int64)
    self_v = np.arange(n, dtype=np.int64)
    u = np.concatenate([edges_u, self_u])
    v = np.concatenate([edges_v, self_v])
    deg = np.bincount(u, minlength=n).astype(np.float32)
    deg = np.where(deg < 1.0, 1.0, deg)
    vals = (1.0 / deg[u]).astype(np.float32)
    idx = torch.tensor(np.vstack([u, v]), dtype=torch.long, device=device)
    val = torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce()

def run_gnn_sparse(X_full, y_full, train_mask, val_mask, test_mask,
                   edges_u, edges_v, out_prefix):
    dev = get_device()
    n, d = X_full.shape

    X = torch.tensor(X_full, dtype=torch.float32, device=dev)
    y = torch.tensor(y_full.astype(np.float32), dtype=torch.float32, device=dev)

    tr = torch.tensor(train_mask, dtype=torch.bool, device=dev)
    va = torch.tensor(val_mask, dtype=torch.bool, device=dev)
    te = torch.tensor(test_mask, dtype=torch.bool, device=dev)

    adj = build_sparse_adj(n, edges_u, edges_v, dev)

    model = SparseGraphSAGE(in_dim=d, hidden=cfg.hidden, layers=cfg.gnn_hops, dropout=cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-5)

    pw = compute_pos_weight(y_full[train_mask])
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=dev))

    hist = {"epoch": [], "train_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1
    best_state = None

    for epoch in range(1, cfg.gnn_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X, adj)
        loss = crit(logits[tr], y[tr])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = logits[va]
            v_prob = torch.sigmoid(v_logits).detach().cpu().numpy()
            v_pred = (v_prob >= 0.5).astype(int)
            v_true = y_full[val_mask]
            v_acc = accuracy_score(v_true, v_pred)
            v_f1  = f1_score(v_true, v_pred, average="macro", zero_division=0)

        hist["epoch"].append(epoch)
        hist["train_loss"].append(float(loss.item()))
        hist["val_acc"].append(float(v_acc))
        hist["val_f1"].append(float(v_f1))

        if v_f1 > best_f1:
            best_f1 = v_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"[GNN hops={cfg.gnn_hops}] Epoch {epoch:03d}: loss={loss.item():.4f}, val_acc={v_acc:.4f}, val_f1={v_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X, adj)
        prob_all = torch.sigmoid(logits).detach().cpu().numpy()
        pred_all = (prob_all >= 0.5).astype(int)

    plot_training_curves(
        hist,
        title=f"{out_prefix}: GNN Training Curves (Sparse GraphSAGE, hops={cfg.gnn_hops})",
        save_path=str(Path(cfg.out_dir) / f"{out_prefix}_gnn_training_curves.png"),
    )

    y_true = y_full[test_mask]
    y_pred = pred_all[test_mask]
    y_prob = prob_all[test_mask]
    return y_true, y_pred, y_prob, pred_all, prob_all


# ==========================================================
# StackOverflow (no comments endpoint)
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


# ==========================================================
# WikiVote (SNAP)
# ==========================================================

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
# Unified runner
# ==========================================================

def run_dataset_pipeline(user_df: pd.DataFrame, graph: Dict, meta: Dict, out_prefix: str):
    out_dir = cfg.out_dir
    save_json(meta, str(Path(out_dir) / f"{out_prefix}_meta.json"))

    feature_cols = meta["feature_cols_user"]
    X_full = user_df[feature_cols].values.astype(np.float32)
    y_full = user_df["y"].values.astype(int)

    n = len(user_df)
    m_train, m_val, m_test = split_masks(n, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)

    X_full, mu, sd = standardize_from_train(X_full, m_train)
    np.save(str(Path(out_dir) / f"{out_prefix}_mu.npy"), mu)
    np.save(str(Path(out_dir) / f"{out_prefix}_sd.npy"), sd)

    # FULL meta mask
    meta_full = user_df["is_meta"].values.astype(bool)
    # ✅ TEST-only meta mask (FIX)
    meta_test = meta_full[m_test]

    Xtr, ytr = X_full[m_train], y_full[m_train]
    Xva, yva = X_full[m_val], y_full[m_val]
    Xte, yte = X_full[m_test], y_full[m_test]

    results = []

    print(f"\n=== {out_prefix}: XGBoost (tabular) ===")
    pred_xgb, prob_xgb = run_xgboost(Xtr, ytr, Xva, yva, Xte, yte)
    if pred_xgb is not None:
        r = evaluate_and_save("xgboost", yte, pred_xgb, prob_xgb, out_prefix, out_dir, meta_test)
        results.append(r)
        print(r)

    print(f"\n=== {out_prefix}: MLP (tabular) ===")
    pred_all_mlp, prob_all_mlp = run_mlp(Xtr, ytr, Xva, yva, X_full, out_prefix)
    r = evaluate_and_save("mlp", yte, pred_all_mlp[m_test], prob_all_mlp[m_test], out_prefix, out_dir, meta_test)
    results.append(r)
    print(r)

    print(f"\n=== {out_prefix}: GNN (Sparse GraphSAGE, hops={cfg.gnn_hops}) ===")
    edges_u = graph["edges_u"]
    edges_v = graph["edges_v"]
    y_true, y_pred, y_prob, pred_all_gnn, prob_all_gnn = run_gnn_sparse(
        X_full=X_full,
        y_full=y_full,
        train_mask=m_train,
        val_mask=m_val,
        test_mask=m_test,
        edges_u=edges_u,
        edges_v=edges_v,
        out_prefix=out_prefix
    )
    r = evaluate_and_save("gnn", y_true, y_pred, y_prob, out_prefix, out_dir, meta_test)
    results.append(r)
    print(r)

    df_res = pd.DataFrame(results)
    df_res.to_csv(str(Path(out_dir) / f"{out_prefix}_results_summary.csv"), index=False)

    tmp = user_df.copy()
    tmp["type3"] = 0
    tmp.loc[tmp["is_influencer"] == 1, "type3"] = 1
    tmp.loc[(tmp["is_influencer"] == 0) & (tmp["is_meta"] == 1), "type3"] = 2
    tmp.groupby("type3")[feature_cols].mean().to_csv(str(Path(out_dir) / f"{out_prefix}_avg_features_by_type3.csv"))

    print(f"\n[INFO] Saved all outputs under: {out_dir}")
    return df_res


def main():
    set_seed(cfg.seed)
    ensure_dirs()

    dev = get_device()
    print(f"Device: {dev} | HAS_XGB={HAS_XGB} | HAS_REQUESTS={HAS_REQUESTS}")

    print("\n==============================")
    print("PHASE 2A: STACKOVERFLOW")
    print("==============================")
    try:
        user_df_so, graph_so, meta_so = build_stackoverflow_dataset()
        print("[STACKOVERFLOW DATA SUMMARY]")
        print(json.dumps(meta_so, indent=2))
        out_prefix = f"phase2_stackoverflow_{cfg.so_tag}".replace("-", "_")
        run_dataset_pipeline(user_df_so, graph_so, meta_so, out_prefix)
    except Exception as e:
        print("\n[STACKOVERFLOW ERROR]")
        print(str(e))

    print("\n==============================")
    print("PHASE 2B: SNAP WIKI-VOTE")
    print("==============================")
    try:
        user_df_wv, graph_wv, meta_wv = build_wikivote_dataset()
        print("[WIKIVOTE DATA SUMMARY]")
        print(json.dumps(meta_wv, indent=2))
        out_prefix = "phase2_wikivote"
        run_dataset_pipeline(user_df_wv, graph_wv, meta_wv, out_prefix)
    except Exception as e:
        print("\n[WIKIVOTE ERROR]")
        print(str(e))

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
