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

# ---------------- Optional deps ----------------
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

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv, to_hetero
    from torch_geometric.transforms import ToUndirected
    HAS_PYG = True
except Exception:
    HAS_PYG = False


# ==========================================================
# Config
# ==========================================================

@dataclass
class CFG:
    # Meta (robust) knobs
    meta_frac: float = 0.05              # metas = 5% of all users (selected from candidates)
    meta_rep_max_quantile: float = 0.60  # meta must be <= 60th percentile of rep_proxy (tabular-similar)
    meta_min_influ_receivers: int = 2    # meta must have >= 2 distinct influencer interactions received

    # Data collection
    site: str = "stackoverflow"
    tag: str = "machine-learning"        # try: "python", "pytorch", "javascript", ...
    pagesize: int = 100
    max_pages_questions: int = 6
    max_pages_answers: int = 6
    max_pages_comments: int = 6

    # API throttling
    sleep_s: float = 0.2
    api_key: Optional[str] = None

    # Influencer labeling
    influencer_top_pct: float = 0.10     # top 10% by rep_proxy are "influencers"

    # split
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # models
    mlp_epochs: int = 60
    gnn_epochs: int = 80
    lr: float = 1e-3
    hidden: int = 64
    dropout: float = 0.2

    # GPU
    use_cuda_if_available: bool = True

    # output
    out_dir: str = "results_phase2"
    cache_dir: str = "cache_phase2"


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

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_confusion(cm, labels, title, save_path: Path):
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

def plot_roc(y_true, y_score, title, save_path: Path):
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

def split_masks(n: int, train_frac: float, val_frac: float, test_frac: float, seed: int):
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


# ==========================================================
# StackExchange API
# ==========================================================

def api_get(endpoint: str, params: Dict) -> Dict:
    if not HAS_REQUESTS:
        raise RuntimeError("requests not installed. Run: pip install requests")

    base = "https://api.stackexchange.com/2.3"
    url = base + endpoint

    params = dict(params)
    params["site"] = cfg.site
    params["pagesize"] = cfg.pagesize
    if cfg.api_key:
        params["key"] = cfg.api_key

    r = requests.get(url, params=params, headers={"User-Agent": "phase2-gnn-benchmark/1.0"})
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")

    data = r.json()
    if "backoff" in data:
        time.sleep(float(data["backoff"]) + cfg.sleep_s)

    time.sleep(cfg.sleep_s)
    return data

def fetch_questions_by_tag() -> List[Dict]:
    cache_path = Path(cfg.cache_dir) / f"questions_{cfg.site}_{cfg.tag}.json"
    if cache_path.exists():
        return load_json(cache_path)

    all_items = []
    for page in range(1, cfg.max_pages_questions + 1):
        data = api_get(
            "/questions",
            params={
                "page": page,
                "order": "desc",
                "sort": "votes",
                "tagged": cfg.tag,
            },
        )
        all_items.extend(data.get("items", []))
        if not data.get("has_more", False):
            break

    save_json(all_items, cache_path)
    return all_items

def fetch_answers_for_questions(question_ids: List[int]) -> List[Dict]:
    cache_path = Path(cfg.cache_dir) / f"answers_{cfg.site}_{cfg.tag}.json"
    if cache_path.exists():
        return load_json(cache_path)

    all_items = []
    chunk = 100
    for i in range(0, len(question_ids), chunk):
        ids = ";".join(map(str, question_ids[i:i + chunk]))
        for page in range(1, cfg.max_pages_answers + 1):
            data = api_get(
                f"/questions/{ids}/answers",
                params={
                    "page": page,
                    "order": "desc",
                    "sort": "votes",
                },
            )
            all_items.extend(data.get("items", []))
            if not data.get("has_more", False):
                break

    save_json(all_items, cache_path)
    return all_items

def fetch_comments_for_posts(post_ids: List[int]) -> List[Dict]:
    cache_path = Path(cfg.cache_dir) / f"comments_{cfg.site}_{cfg.tag}.json"
    if cache_path.exists():
        return load_json(cache_path)

    all_items = []
    chunk = 100
    for i in range(0, len(post_ids), chunk):
        ids = ";".join(map(str, post_ids[i:i + chunk]))
        for page in range(1, cfg.max_pages_comments + 1):
            data = api_get(
                f"/posts/{ids}/comments",
                params={
                    "page": page,
                    "order": "desc",
                    "sort": "creation",
                },
            )
            all_items.extend(data.get("items", []))
            if not data.get("has_more", False):
                break

    save_json(all_items, cache_path)
    return all_items


# ==========================================================
# Build real dataset + hetero graph
# ==========================================================

def build_real_dataset() -> Tuple[pd.DataFrame, "HeteroData", Dict]:
    if not HAS_PYG:
        raise RuntimeError("torch-geometric not installed.")

    questions = fetch_questions_by_tag()
    q_ids = [q["question_id"] for q in questions]

    answers = fetch_answers_for_questions(q_ids)

    post_ids = sorted(set([q["question_id"] for q in questions] + [a["answer_id"] for a in answers]))
    comments = fetch_comments_for_posts(post_ids)

    def owner_uid(obj):
        return obj.get("owner", {}).get("user_id", None)

    # ---------------- Users ----------------
    user_ids = set()
    for q in questions:
        u = owner_uid(q)
        if u is not None:
            user_ids.add(u)
    for a in answers:
        u = owner_uid(a)
        if u is not None:
            user_ids.add(u)
    for c in comments:
        u = owner_uid(c)
        if u is not None:
            user_ids.add(u)

    user_ids = sorted(user_ids)
    user2idx = {u: i for i, u in enumerate(user_ids)}

    # ---------------- Posts (questions + answers) ----------------
    posts = []
    post2idx = {}

    def add_post(post_id, ptype, score, owner_id, parent_qid=None):
        if post_id in post2idx:
            return
        post2idx[post_id] = len(posts)
        posts.append({
            "post_id": post_id,
            "ptype": ptype,
            "score": float(score),
            "owner_id": owner_id,
            "parent_qid": parent_qid
        })

    for q in questions:
        add_post(q["question_id"], "question", q.get("score", 0), owner_uid(q), None)

    for a in answers:
        add_post(a["answer_id"], "answer", a.get("score", 0), owner_uid(a), a.get("question_id", None))

    # ---------------- Comments ----------------
    comments_list = []
    comment2idx = {}

    def add_comment(comment_id, score, owner_id, post_id):
        if comment_id in comment2idx:
            return
        comment2idx[comment_id] = len(comments_list)
        comments_list.append({
            "comment_id": comment_id,
            "score": float(score),
            "owner_id": owner_id,
            "post_id": post_id
        })

    for c in comments:
        add_comment(c["comment_id"], c.get("score", 0), owner_uid(c), c.get("post_id", None))

    # ---------------- User features + interactions ----------------
    u_stats = {
        u: dict(
            num_questions=0,
            num_answers=0,
            num_comments=0,
            sum_post_score=0.0,
            sum_comment_score=0.0,
            distinct_posts_interacted=set(),
            distinct_users_interacted=set(),
            distinct_influencers_interacted_with_me=set(),
        )
        for u in user_ids
    }

    def record_actor_to_owner(actor_uid, owner_uid):
        if actor_uid is None or owner_uid is None:
            return
        if actor_uid == owner_uid:
            return
        if actor_uid not in u_stats or owner_uid not in u_stats:
            return
        u_stats[actor_uid]["distinct_users_interacted"].add(owner_uid)

    # post ownership stats
    for p in posts:
        uid = p["owner_id"]
        if uid is None or uid not in u_stats:
            continue
        if p["ptype"] == "question":
            u_stats[uid]["num_questions"] += 1
        else:
            u_stats[uid]["num_answers"] += 1
        u_stats[uid]["sum_post_score"] += float(p["score"])

    # comment stats + comment->post-owner interaction
    for cm in comments_list:
        uid = cm["owner_id"]
        if uid is None or uid not in u_stats:
            continue
        u_stats[uid]["num_comments"] += 1
        u_stats[uid]["sum_comment_score"] += float(cm["score"])

        pid = cm.get("post_id", None)
        if pid is not None:
            u_stats[uid]["distinct_posts_interacted"].add(pid)

        if pid in post2idx:
            owner = posts[post2idx[pid]]["owner_id"]
            record_actor_to_owner(uid, owner)

    # answering a question -> interacts with question owner
    q_owner = {q["question_id"]: owner_uid(q) for q in questions}
    for a in answers:
        actor = owner_uid(a)
        qid = a.get("question_id", None)
        owner = q_owner.get(qid, None)
        record_actor_to_owner(actor, owner)

    # build dataframe with rep_proxy
    rows = []
    for u in user_ids:
        s = u_stats[u]
        rep_proxy = (
            5.0 * s["sum_post_score"] +
            2.0 * s["sum_comment_score"] +
            3.0 * s["num_answers"] +
            2.0 * s["num_questions"]
        )
        rows.append({
            "user_id": u,
            "num_questions": s["num_questions"],
            "num_answers": s["num_answers"],
            "num_comments": s["num_comments"],
            "sum_post_score": s["sum_post_score"],
            "sum_comment_score": s["sum_comment_score"],
            "distinct_users_interacted": len(s["distinct_users_interacted"]),
            "distinct_posts_interacted": len(s["distinct_posts_interacted"]),
            "rep_proxy": rep_proxy,
        })

    user_df = pd.DataFrame(rows)

    # influencers: top percentile by rep_proxy
    thr = np.quantile(user_df["rep_proxy"].values, 1.0 - cfg.influencer_top_pct)
    user_df["is_influencer"] = (user_df["rep_proxy"] >= thr).astype(int)
    influencer_set = set(user_df.loc[user_df["is_influencer"] == 1, "user_id"].tolist())

    # collect influencer->target interactions received
    def add_influencer_to_target(influ_uid, target_uid):
        if influ_uid is None or target_uid is None:
            return
        if influ_uid not in influencer_set:
            return
        if target_uid not in u_stats:
            return
        u_stats[target_uid]["distinct_influencers_interacted_with_me"].add(influ_uid)

    # influencer answers -> question owner
    for a in answers:
        influ = owner_uid(a)
        qid = a.get("question_id", None)
        target = q_owner.get(qid, None)
        add_influencer_to_target(influ, target)

    # influencer comments -> post owner
    for cm in comments_list:
        influ = cm["owner_id"]
        pid = cm.get("post_id", None)
        if pid in post2idx:
            target = posts[post2idx[pid]]["owner_id"]
            add_influencer_to_target(influ, target)

    # robust meta assignment
    user_df["influ_interactions_received"] = user_df["user_id"].apply(
        lambda uid: len(u_stats[uid]["distinct_influencers_interacted_with_me"])
    ).astype(int)

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

    # binary target
    user_df["y"] = ((user_df["is_influencer"] == 1) | (user_df["is_meta"] == 1)).astype(int)

    # ---------------- HeteroData ----------------
    data = HeteroData()

    feature_cols = [
        "num_questions", "num_answers", "num_comments",
        "sum_post_score", "sum_comment_score",
        "distinct_users_interacted", "distinct_posts_interacted",
        "rep_proxy",
    ]

    data["user"].x = torch.tensor(user_df[feature_cols].values, dtype=torch.float32)
    data["user"].y = torch.tensor(user_df["y"].values, dtype=torch.long)

    # post features: [score, is_question, is_answer]
    p_feats = []
    for p in posts:
        score = float(p["score"])
        is_q = 1.0 if p["ptype"] == "question" else 0.0
        is_a = 1.0 if p["ptype"] == "answer" else 0.0
        p_feats.append([score, is_q, is_a])
    data["post"].x = torch.tensor(p_feats, dtype=torch.float32)

    # comment features: [score]
    c_feats = [[float(c["score"])] for c in comments_list]
    data["comment"].x = torch.tensor(c_feats, dtype=torch.float32)

    # edges
    src_asked, dst_asked = [], []
    src_ans, dst_ans = [], []

    for p in posts:
        uid = p["owner_id"]
        if uid is None or uid not in user2idx:
            continue
        if p["ptype"] == "question":
            src_asked.append(user2idx[uid])
            dst_asked.append(post2idx[p["post_id"]])
        else:
            src_ans.append(user2idx[uid])
            dst_ans.append(post2idx[p["post_id"]])

    data[("user", "asked", "post")].edge_index = (
        torch.tensor([src_asked, dst_asked], dtype=torch.long)
        if len(src_asked) else torch.empty((2, 0), dtype=torch.long)
    )
    data[("user", "answered", "post")].edge_index = (
        torch.tensor([src_ans, dst_ans], dtype=torch.long)
        if len(src_ans) else torch.empty((2, 0), dtype=torch.long)
    )

    # user -> comment
    src_uc, dst_uc = [], []
    for c in comments_list:
        uid = c["owner_id"]
        if uid is None or uid not in user2idx:
            continue
        src_uc.append(user2idx[uid])
        dst_uc.append(comment2idx[c["comment_id"]])

    data[("user", "commented", "comment")].edge_index = (
        torch.tensor([src_uc, dst_uc], dtype=torch.long)
        if len(src_uc) else torch.empty((2, 0), dtype=torch.long)
    )

    # comment -> post
    src_cp, dst_cp = [], []
    for c in comments_list:
        pid = c.get("post_id", None)
        if pid is None or pid not in post2idx:
            continue
        src_cp.append(comment2idx[c["comment_id"]])
        dst_cp.append(post2idx[pid])

    data[("comment", "on", "post")].edge_index = (
        torch.tensor([src_cp, dst_cp], dtype=torch.long)
        if len(src_cp) else torch.empty((2, 0), dtype=torch.long)
    )

    # user -> user interacts (actor -> owner)
    src_uu, dst_uu = [], []
    for u in user_ids:
        uidx = user2idx[u]
        for v in u_stats[u]["distinct_users_interacted"]:
            if v in user2idx:
                src_uu.append(uidx)
                dst_uu.append(user2idx[v])

    data[("user", "interacts", "user")].edge_index = (
        torch.tensor([src_uu, dst_uu], dtype=torch.long)
        if len(src_uu) else torch.empty((2, 0), dtype=torch.long)
    )

    # undirected helps message passing
    data = ToUndirected()(data)

    # masks
    m_train, m_val, m_test = split_masks(len(user_df), cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)
    data["user"].train_mask = torch.tensor(m_train, dtype=torch.bool)
    data["user"].val_mask = torch.tensor(m_val, dtype=torch.bool)
    data["user"].test_mask = torch.tensor(m_test, dtype=torch.bool)

    meta = {
        "n_users": len(user_ids),
        "n_posts": len(posts),
        "n_comments": len(comments_list),
        "n_edges_user_interacts_user": int(len(src_uu)),
        "influencer_threshold_rep_proxy": float(thr),
        "label_stats": {
            "positives": int(user_df["y"].sum()),
            "negatives": int((user_df["y"] == 0).sum()),
            "influencers": int(user_df["is_influencer"].sum()),
            "metas": int(user_df["is_meta"].sum()),
        },
        "feature_cols_user": feature_cols,
    }

    return user_df, data, meta


# ==========================================================
# Baselines
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

def run_xgboost(train_X, train_y, val_X, val_y, test_X):
    if not HAS_XGB:
        return None, None

    model = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.08,
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

def run_mlp(train_X, train_y, val_X, val_y, test_X, out_prefix: str):
    dev = get_device()
    Xtr = torch.tensor(train_X, dtype=torch.float32, device=dev)
    ytr = torch.tensor(train_y, dtype=torch.float32, device=dev)
    Xva = torch.tensor(val_X, dtype=torch.float32, device=dev)
    Xte = torch.tensor(test_X, dtype=torch.float32, device=dev)

    model = TabularMLP(in_dim=train_X.shape[1], hidden=cfg.hidden, dropout=cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-5)
    crit = nn.BCEWithLogitsLoss()

    hist = {"epoch": [], "train_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1
    best_state = None

    for epoch in range(1, cfg.mlp_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = crit(logits, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_prob = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_pred = (val_prob >= 0.5).astype(int)
            v_acc = accuracy_score(val_y, val_pred)
            v_f1 = f1_score(val_y, val_pred, average="macro")

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
        test_logits = model(Xte)
        test_prob = torch.sigmoid(test_logits).detach().cpu().numpy()
        test_pred = (test_prob >= 0.5).astype(int)

    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_acc"], label="val_acc")
    plt.plot(hist["epoch"], hist["val_f1"], label="val_f1")
    plt.xlabel("Epoch")
    plt.title("MLP Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(cfg.out_dir) / f"{out_prefix}_mlp_training_curves.png", dpi=160)
    plt.close()

    return test_pred, test_prob


# ==========================================================
# GNN
# ==========================================================

class BaseSAGE(nn.Module):
    def __init__(self, hidden=64, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), hidden)
        self.dropout = dropout
        self.lin = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)

def run_gnn(hetero: "HeteroData", out_prefix: str):
    dev = get_device()
    hetero = hetero.to(dev)

    base = BaseSAGE(hidden=cfg.hidden, dropout=cfg.dropout).to(dev)
    model = to_hetero(base, hetero.metadata(), aggr="sum").to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-5)
    crit = nn.BCEWithLogitsLoss()

    train_mask = hetero["user"].train_mask
    val_mask = hetero["user"].val_mask
    test_mask = hetero["user"].test_mask
    y = hetero["user"].y.float()

    hist = {"epoch": [], "train_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1
    best_state = None

    for epoch in range(1, cfg.gnn_epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(hetero.x_dict, hetero.edge_index_dict)["user"]
        loss = crit(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_val = model(hetero.x_dict, hetero.edge_index_dict)["user"][val_mask]
            val_prob = torch.sigmoid(out_val).detach().cpu().numpy()
            val_pred = (val_prob >= 0.5).astype(int)
            val_true = hetero["user"].y[val_mask].detach().cpu().numpy()
            v_acc = accuracy_score(val_true, val_pred)
            v_f1 = f1_score(val_true, val_pred, average="macro")

        hist["epoch"].append(epoch)
        hist["train_loss"].append(float(loss.item()))
        hist["val_acc"].append(float(v_acc))
        hist["val_f1"].append(float(v_f1))

        if v_f1 > best_f1:
            best_f1 = v_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"[GNN] Epoch {epoch:03d}: loss={loss.item():.4f}, val_acc={v_acc:.4f}, val_f1={v_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out_test = model(hetero.x_dict, hetero.edge_index_dict)["user"][test_mask]
        test_prob = torch.sigmoid(out_test).detach().cpu().numpy()
        test_pred = (test_prob >= 0.5).astype(int)
        test_true = hetero["user"].y[test_mask].detach().cpu().numpy()

    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_acc"], label="val_acc")
    plt.plot(hist["epoch"], hist["val_f1"], label="val_f1")
    plt.xlabel("Epoch")
    plt.title("GNN Training Curves (Hetero GraphSAGE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(cfg.out_dir) / f"{out_prefix}_gnn_training_curves.png", dpi=160)
    plt.close()

    return test_true, test_pred, test_prob


# ==========================================================
# Eval + save
# ==========================================================

def evaluate_and_save(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, prefix: str):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")
    cm = confusion_matrix(y_true, y_pred)

    plot_confusion(
        cm,
        labels=["normal(0)", "influence(1)"],
        title=f"{name} Confusion Matrix",
        save_path=Path(cfg.out_dir) / f"{prefix}_{name}_cm.png",
    )
    plot_roc(
        y_true,
        y_score,
        title=f"{name} ROC",
        save_path=Path(cfg.out_dir) / f"{prefix}_{name}_roc.png",
    )

    rep = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return {
        "model": name,
        "accuracy": float(acc),
        "macro_f1": float(f1m),
        "auc": float(auc),
        "precision_pos": float(rep["1"]["precision"]) if "1" in rep else None,
        "recall_pos": float(rep["1"]["recall"]) if "1" in rep else None,
        "f1_pos": float(rep["1"]["f1-score"]) if "1" in rep else None,
    }

def meta_subset_metrics(name: str, y_true_full: np.ndarray, y_pred_full: np.ndarray, meta_mask: np.ndarray):
    if meta_mask.sum() == 0:
        print(f"[WARN] No meta nodes in test split for {name}.")
        return

    y_true_meta = y_true_full[meta_mask]
    y_pred_meta = y_pred_full[meta_mask]

    tp = ((y_pred_meta == 1) & (y_true_meta == 1)).sum()
    fp = ((y_pred_meta == 1) & (y_true_meta == 0)).sum()
    fn = ((y_pred_meta == 0) & (y_true_meta == 1)).sum()

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * prec * rec / max(1e-9, (prec + rec))

    print(f"\n{name} – META subset:")
    print(f"  #meta_test={meta_mask.sum()}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")


# ==========================================================
# Main
# ==========================================================

def main():
    set_seed(cfg.seed)
    ensure_dirs()

    print("=== Phase 2: REAL DATA (StackOverflow) ===")
    print(f"Tag: {cfg.tag}, site: {cfg.site}, device: {get_device()}")

    user_df, hetero, meta = build_real_dataset()
    save_json(meta, Path(cfg.out_dir) / "phase2_meta_summary.json")

    print("\n[DATA SUMMARY]")
    print(json.dumps(meta, indent=2))

    feature_cols = meta["feature_cols_user"]
    X = user_df[feature_cols].values.astype(np.float32)
    y = user_df["y"].values.astype(int)

    train_mask = hetero["user"].train_mask.cpu().numpy()
    val_mask = hetero["user"].val_mask.cpu().numpy()
    test_mask = hetero["user"].test_mask.cpu().numpy()

    # meta subset within TEST only
    meta_test_mask = (user_df["is_meta"].values.astype(bool)) & (test_mask)

    Xtr, ytr = X[train_mask], y[train_mask]
    Xva, yva = X[val_mask], y[val_mask]
    Xte, yte = X[test_mask], y[test_mask]

    prefix = f"phase2_{cfg.site}_{cfg.tag}".replace("-", "_")
    results = []

    # ---------- XGBoost ----------
    pred_xgb, prob_xgb = None, None
    if HAS_XGB:
        print("\n=== XGBoost (tabular) ===")
        pred_xgb, prob_xgb = run_xgboost(Xtr, ytr, Xva, yva, Xte)
        if pred_xgb is not None:
            r = evaluate_and_save("xgboost", yte, pred_xgb, prob_xgb, prefix)
            results.append(r)
            print(r)
    else:
        print("\n[WARN] xgboost not installed -> skipping XGBoost baseline.")

    # ---------- MLP ----------
    print("\n=== MLP (tabular) ===")
    pred_mlp, prob_mlp = run_mlp(Xtr, ytr, Xva, yva, Xte, prefix)
    r = evaluate_and_save("mlp", yte, pred_mlp, prob_mlp, prefix)
    results.append(r)
    print(r)

    # ---------- GNN ----------
    print("\n=== GNN (hetero GraphSAGE) ===")
    y_true_gnn, pred_gnn, prob_gnn = run_gnn(hetero, prefix)
    r = evaluate_and_save("gnn", y_true_gnn, pred_gnn, prob_gnn, prefix)
    results.append(r)
    print(r)

    # ---------- Meta-subset metrics ----------
    # For tabular models: yte is aligned with test rows
    if pred_xgb is not None and prob_xgb is not None:
        meta_subset_metrics("XGBoost", yte, pred_xgb, meta_test_mask[test_mask])
    meta_subset_metrics("MLP", yte, pred_mlp, meta_test_mask[test_mask])

    # For GNN: y_true_gnn/pred_gnn aligned with user test_mask order
    meta_subset_metrics("GNN", y_true_gnn, pred_gnn, meta_test_mask[test_mask])

    # Save results
    df_res = pd.DataFrame(results)
    out_csv = Path(cfg.out_dir) / f"{prefix}_results_summary.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved results summary: {out_csv}")

    # avg feature table by internal 3-type (for slides)
    df = user_df.copy()
    df["type3"] = 0
    df.loc[df["is_influencer"] == 1, "type3"] = 1
    df.loc[(df["is_influencer"] == 0) & (df["is_meta"] == 1), "type3"] = 2
    grp = df.groupby("type3")[feature_cols].mean()
    grp.to_csv(Path(cfg.out_dir) / f"{prefix}_avg_features_by_type3.csv")
    print("[INFO] Saved avg feature table by type3 (0 normal,1 infl,2 meta).")

if __name__ == "__main__":
    main()
