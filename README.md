# Graph Neural Networks for Hierarchical Influence Detection

**Author:** Ashik Ali Shaik  
**Focus:** GNNs · Graph ML · GPU Acceleration · Influence Detection  

---

## 🔎 One-Line Summary
A research-grade evaluation showing **why Graph Neural Networks outperform feature-only models** for detecting *influencers of influencers*, with **CPU vs NVIDIA GPU performance analysis** on synthetic and real-world graphs.

---

## 📊 Results Snapshot 

| Dataset        | Model        | Accuracy | Macro-F1 | ROC-AUC | GPU Speedup |
|---------------|-------------|----------|----------|---------|-------------|
| Synthetic     | GNN (GCN)    | 0.998    | 0.998    | 1.000   | **75×**     |
| StackOverflow | GraphSAGE   | 0.816    | 0.780    | 0.901   | **6–7×**    |
| WikiVote      | GraphSAGE   | 0.901    | 0.849    | 0.942   | **12–16×**  |
| Baseline      | XGBoost     | Strong   | Lower    | High    | CPU only    |

**Key takeaway:**  
When influence is *structural and multi-hop*, **GNNs win** — and GPUs scale best as depth and hops increase.

---

## 🧠 What This Project Demonstrates
- Why **structure-aware learning** matters
- When **feature-only ML fails**
- How **GNN depth & hops** affect accuracy
- Why **NVIDIA GPUs accelerate sparse graph workloads**

---

## ▶️ How to Run (Quick Start)

```bash
# 1. Create environment
conda create -n gnn python=3.10 -y
conda activate gnn

# 2. Install dependencies
pip install torch torch-geometric scikit-learn matplotlib xgboost pandas numpy

# 3. Run synthetic experiments
python project/phase1_synthetic/train_gnn.py

# 4. Run real-world experiments
python project/phase2_real/train_stackoverflow.py
python project/phase2_real/train_wikivote.py
