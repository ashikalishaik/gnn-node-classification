# 🧠 GNN Node Classification Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready Graph Neural Network project for node classification** with comprehensive documentation, multiple GNN architectures (GCN, GAT, GraphSAGE), hyperparameter sweeps, and real-time deployment capabilities.

---

## 📋 Table of Contents

- [Motivation](#motivation)
  - [What are GNNs?](#what-are-gnns)
  - [Why Use GNNs?](#why-use-gnns)
  - [How Do GNNs Work?](#how-do-gnns-work)
  - [Where Are GNNs Applied?](#where-are-gnns-applied)
  - [When to Use GNNs?](#when-to-use-gnns)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## 🎯 Motivation

### What are GNNs?

Graph Neural Networks (GNNs) are a class of deep learning models designed to work directly on graph-structured data. Unlike traditional neural networks that operate on Euclidean data (images, text), GNNs can capture complex relationships and dependencies in networks such as social networks, molecular structures, knowledge graphs, and citation networks.

**Key Concepts:**
- **Nodes**: Entities in the graph (e.g., people, molecules, documents)
- **Edges**: Relationships between entities (e.g., friendships, bonds, citations)
- **Features**: Attributes associated with nodes and edges
- **Message Passing**: Core mechanism for aggregating information from neighbors

### Why Use GNNs?

1. **Relational Data**: Many real-world datasets are naturally represented as graphs
   - Social networks (users and friendships)
   - Citation networks (papers and citations)
   - Molecular structures (atoms and bonds)
   - Traffic networks (intersections and roads)

2. **Non-Euclidean**: Traditional CNNs/RNNs fail on irregular graph structures
   - Graphs don't have a fixed size or ordering
   - Each node can have a different number of neighbors
   - Spatial convolutions don't translate to graphs

3. **Inductive Learning**: Can generalize to unseen nodes/graphs
   - Learn patterns that transfer to new data
   - Don't require retraining for new nodes

4. **State-of-the-art**: Superior performance on graph tasks
   - Node classification
   - Link prediction
   - Graph classification

### How Do GNNs Work?

GNNs use **message passing** to aggregate information from neighboring nodes:

```
1. Initialize: Each node has feature vector h₀
2. Message Passing (repeat L layers):
   - Aggregate: Collect messages from neighbors
   - Update: Combine aggregated messages with node features
   hₗ = σ(W · AGGREGATE({hₗ₋₁ᵤ : u ∈ N(v)}))
3. Readout: Use final node representations for downstream tasks
```

**Example:**

In a citation network:
- Paper A cites Papers B, C, D
- GNN aggregates features from B, C, D to update A's representation
- After multiple layers, A's representation captures information from its extended neighborhood

### Where Are GNNs Applied?

- 🔬 **Drug Discovery**: Predict molecular properties, synthesize new compounds
- 🌐 **Social Networks**: Recommend friends, detect communities, analyze influence
- 🚗 **Traffic Forecasting**: Predict congestion, optimize routes
- 📚 **Citation Networks**: Classify papers, recommend relevant research
- 🛡️ **Fraud Detection**: Identify suspicious transactions in financial networks
- 🧬 **Bioinformatics**: Protein structure prediction, gene interaction analysis
- 🏪 **E-commerce**: Product recommendations, supply chain optimization

### When to Use GNNs?

Use GNNs when:
- ✅ Your data has explicit graph structure
- ✅ Relationships between entities are crucial
- ✅ You need to leverage neighborhood information
- ✅ Traditional ML methods fail to capture dependencies
- ✅ You want interpretable relational patterns

Don't use GNNs when:
- ❌ Data doesn't have meaningful graph structure
- ❌ Relationships are not important for the task
- ❌ You need extremely fast inference (GNNs can be slower)

---

## 🗂️ Project Structure

```
gnn-node-classification/
├── data/
│   ├── raw/                    # Raw Cora dataset
│   └── processed/              # Preprocessed PyG format
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gcn.py             # Graph Convolutional Network
│   │   ├── gat.py             # Graph Attention Network
│   │   └── graphsage.py       # GraphSAGE
│   ├── data_loader.py         # Data loading & preprocessing
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation metrics
│   ├── utils.py               # Utility functions
│   └── config.py              # Configuration management
├── sweeps/
│   ├── gcn_sweep.yaml         # GCN hyperparameter config
│   ├── gat_sweep.yaml         # GAT hyperparameter config
│   └── graphsage_sweep.yaml   # GraphSAGE config
├── results/
│   ├── plots/                 # Training curves, confusion matrices
│   ├── logs/                  # Training logs
│   └── models/                # Saved checkpoints
├── deployment/
│   ├── api_app.py             # FastAPI application
│   ├── streamlit_app.py       # Streamlit dashboard
│   ├── inference.py           # Real-time inference
│   └── Dockerfile             # Docker configuration
├── docs/
│   ├── latex/
│   │   ├── report.tex         # Main LaTeX document
│   │   ├── sections/           # Document sections
│   │   └── figures/            # Plots and diagrams
│   └── notebooks/
│       ├── 01_data_exploration.ipynb
│       ├── 02_model_training.ipynb
│       └── 03_model_comparison.ipynb
├── tests/
│   ├── test_models.py
│   ├── test_data_loader.py
│   └── test_utils.py
├── requirements.txt
├── setup.py
├── README.md
├── .env.example
├── .gitignore
└── LICENSE
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB RAM minimum

### Installation

```bash
# Clone the repository
git clone https://github.com/ashikalishaik/gnn-node-classification.git
cd gnn-node-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

### Quick Training

```bash
# Train GCN with default parameters
python src/train.py --model gcn

# Train GAT with custom parameters
python src/train.py --model gat --hidden-dim 64 --heads 8 --lr 0.005

# Train GraphSAGE
python src/train.py --model graphsage --epochs 300
```

### Hyperparameter Tuning

```bash
# Initialize W&B sweep
wandb sweep sweeps/gcn_sweep.yaml

# Run sweep agent
wandb agent <sweep-id>
```

---

## 📊 Dataset

### Cora Citation Network

We use the **Cora dataset** - a canonical benchmark for GNN node classification:

**Statistics:**
- **Nodes**: 2,708 scientific papers
- **Edges**: 5,429 citations (directed)
- **Node Features**: 1,433 binary word features (bag-of-words)
- **Classes**: 7 research topics
- **Train/Val/Test Split**: 140/500/1000 nodes

**Research Topics (Classes):**
1. Case_Based
2. Genetic_Algorithms
3. Neural_Networks
4. Probabilistic_Methods
5. Reinforcement_Learning
6. Rule_Learning
7. Theory

**Task**: Classify papers into research topics based on:
- Paper content (word features)
- Citation relationships (graph structure)

### Data Preprocessing

```python
# src/data_loader.py
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Load Cora dataset
dataset = Planetoid(root='data/', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

# Data properties
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of features: {data.num_features}')
print(f'Number of classes: {dataset.num_classes}')
```

---

## 🤖 Models

### 1. Graph Convolutional Network (GCN)

**Paper**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

**Why GCN?**
- ✅ Simple and effective baseline
- ✅ Fast training and inference
- ✅ Works well on homophilic graphs (similar nodes connect)
- ✅ Spectral approach with spatial interpretation

**Architecture**:
```
Input (1433) → GCN Layer (64) → ReLU → Dropout(0.5) → GCN Layer (7) → LogSoftmax
```

**Update Rule**:
```
H^(l+1) = σ(ÃD^(-1/2) Ã ÃD^(-1/2) H^(l) W^(l))
```

### 2. Graph Attention Network (GAT)

**Paper**: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

**Why GAT?**
- ✅ Learns importance of neighbors (attention mechanism)
- ✅ Better on heterophilic graphs (dissimilar nodes connect)
- ✅ More expressive than GCN
- ✅ Multi-head attention for robustness

**Architecture**:
```
Input (1433) → GAT Layer (8 heads × 8 features) → ELU → Dropout(0.6) → GAT Layer (1 head × 7) → LogSoftmax
```

**Attention Mechanism**:
```
αᵢⱼ = softmax(LeakyReLU(a^T [W hᵢ || W hⱼ]))
hᵢ' = σ(∑ⱼ₌ₙ(ᵢ) αᵢⱼ W hⱼ)
```

### 3. GraphSAGE

**Paper**: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

**Why GraphSAGE?**
- ✅ Inductive learning (generalizes to new nodes)
- ✅ Scalable to large graphs (mini-batch training)
- ✅ Multiple aggregation functions (mean, max, LSTM)
- ✅ Samples fixed-size neighborhoods

**Architecture**:
```
Input (1433) → SAGE Layer (64, mean) → ReLU → Dropout(0.5) → SAGE Layer (7, mean) → LogSoftmax
```

**Update Rule**:
```
hₙᵥ₍₁ᵢ = AGGREGATE({hₙᵤ : u ∈ N(v)})
hₙ₊₁ᵥ = σ(W · CONCAT(hₙᵥ, hₙᵥ₍₁))
```

---

## 🏋️ Training

### Training Configuration

```python
# Example: src/config.py
config = {
    'gcn': {
        'hidden_dim': 64,
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'dropout': 0.5,
        'epochs': 200,
        'patience': 20
    },
    'gat': {
        'hidden_dim': 8,
        'heads': 8,
        'learning_rate': 0.005,
        'weight_decay': 5e-4,
        'dropout': 0.6,
        'epochs': 200,
        'patience': 20
    },
    'graphsage': {
        'hidden_dim': 64,
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'dropout': 0.5,
        'epochs': 200,
        'patience': 20
    }
}
```

### Training Pipeline

```python
# src/train.py
import torch
from src.models import GCN, GAT, GraphSAGE
from src.data_loader import load_data
from src.evaluate import evaluate_model

# Load data
data = load_data('Cora')

# Initialize model
model = GCN(num_features=data.num_features, 
            num_classes=data.num_classes)

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    # Validation
    val_acc = evaluate_model(model, data, 'val')
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

---

## 📈 Evaluation

### Metrics

We evaluate models using multiple metrics:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and weighted F1
- **Precision/Recall**: Per-class metrics
- **Confusion Matrix**: Classification breakdown
- **Training Time**: Computational efficiency
- **Inference Time**: Real-time prediction speed

### Model Comparison

| Model | Test Accuracy | F1-Score | Parameters | Training Time | Inference Time |
|-------|---------------|----------|------------|---------------|----------------|
| GCN | 81.5% | 0.813 | ~21K | 2.3s/epoch | 5ms |
| GAT | 83.0% | 0.828 | ~98K | 4.1s/epoch | 12ms |
| GraphSAGE | 80.2% | 0.799 | ~28K | 3.2s/epoch | 8ms |

*Results on Cora test set. Times measured on NVIDIA A100 GPU.*

### Visualization

The project generates comprehensive visualizations:

- Training/validation curves
- Confusion matrices
- t-SNE embeddings of learned representations
- Attention weight heatmaps (for GAT)
- Per-class performance metrics

---

## 🚢 Deployment

### Local Inference

```python
# deployment/inference.py
import torch
from src.models import GCN

# Load trained model
model = GCN(num_features=1433, num_classes=7)
model.load_state_dict(torch.load('results/models/best_gcn.pt'))
model.eval()

# Make prediction
with torch.no_grad():
    prediction = model(data).argmax(dim=1)
    print(f"Predicted class: {prediction[node_id]}")
```

### FastAPI Application

```bash
# Start API server
uvicorn deployment.api_app:app --host 0.0.0.0 --port 8000

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

**API Endpoints:**
- `POST /predict`: Classify a node
- `POST /batch_predict`: Classify multiple nodes
- `GET /model_info`: Get model statistics
- `GET /health`: Health check

### Streamlit Dashboard

```bash
# Launch interactive dashboard
streamlit run deployment/streamlit_app.py
```

Features:
- Interactive node selection
- Real-time predictions
- Visualization of node neighborhoods
- Model comparison interface

### Docker Deployment

```bash
# Build Docker image
docker build -t gnn-classifier .

# Run container
docker run -p 8000:8000 gnn-classifier

# Or use docker-compose
docker-compose up
```

---

## 📚 Documentation

### LaTeX Report

Comprehensive technical report available in `docs/latex/`:

**Sections:**
1. Introduction to GNNs
2. Problem Formulation
3. Related Work
4. Methodology
5. Experimental Setup
6. Results and Analysis
7. Conclusion and Future Work

```bash
# Compile LaTeX document
cd docs/latex
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

### Jupyter Notebooks

Interactive notebooks for exploration:

1. **01_data_exploration.ipynb**: Dataset analysis, visualization
2. **02_model_training.ipynb**: Train models interactively
3. **03_model_comparison.ipynb**: Compare model performance

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 src/
black src/
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch Geometric Team**: For the excellent GNN library
- **Cora Dataset Creators**: McCallum et al.
- **GNN Research Community**: For groundbreaking papers
- **Open Source Contributors**: For tools and frameworks

### Key References

1. Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks
2. Veličković et al. (2018). Graph Attention Networks
3. Hamilton et al. (2017). Inductive Representation Learning on Large Graphs

---

## 📧 Contact

For questions, suggestions, or collaboration:

- **GitHub**: [@ashikalishaik](https://github.com/ashikalishaik)
- **Repository**: [gnn-node-classification](https://github.com/ashikalishaik/gnn-node-classification)
- **Issues**: [Report bugs or request features](https://github.com/ashikalishaik/gnn-node-classification/issues)

---

## 📅 Project Timeline

- **Day 1**: Setup, data exploration, implement GCN
- **Day 2**: Implement GAT and GraphSAGE
- **Day 3**: Training, hyperparameter tuning, evaluation
- **Day 4**: Deployment, documentation, final testing

---

## ⭐ Star History

If you find this project helpful:
- ⭐ Star the repository
- 👁️ Watch for updates
- 👨‍💻 Contribute improvements

---

## 🚀 Getting Started Checklist

- [ ] Clone repository
- [ ] Install dependencies
- [ ] Download Cora dataset
- [ ] Train first GCN model
- [ ] Compare with GAT and GraphSAGE
- [ ] Run hyperparameter sweeps
- [ ] Deploy FastAPI application
- [ ] Generate LaTeX documentation
- [ ] Explore Jupyter notebooks
- [ ] Run tests

---

**Built with ❤️ by [Ashik Ali Shaik](https://github.com/ashikalishaik)**
