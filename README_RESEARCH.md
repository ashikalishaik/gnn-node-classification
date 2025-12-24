# Design, Performance, and Hardware-Aware Evaluation of Graph Neural Networks  
## Hierarchical Influence Detection (Influencers of Influencers)

**Author:** Ashik Ali Shaik  
**Date:** December 2025  

---

## Project Overview

Influence detection in social networks is inherently **relational**. A node’s importance often depends not only on its own attributes, but on *who it is connected to*, and *who those neighbors influence in turn*.

This project studies **hierarchical influence detection**, with a specific focus on identifying:

> **Influencers of influencers** — nodes that may have modest direct visibility, yet exert strong *indirect influence* through highly influential neighbors.

To rigorously evaluate this phenomenon, we design a **two-phase empirical framework** and compare **Graph Neural Networks (GNNs)** against strong **feature-only baselines**, while also analyzing **CPU vs NVIDIA GPU performance** in depth.

---

## Research Questions

This repository answers the following questions:

1. **Structural Necessity**  
   When labels depend on *multi-hop graph structure*, do GNNs outperform feature-only models?

2. **Architectural Comparison**  
   How do GCN, GraphSAGE, and GAT compare in accuracy, F1, ROC/AUC, and error modes?

3. **Hardware Performance**  
   How do training and inference times scale on **CPU vs GPU** as:
   - model depth increases  
   - neighborhood hop count increases?

4. **GPU Architecture Insights**  
   Which NVIDIA GPU architectural properties explain observed GNN speedups?

---

## Why Graph Neural Networks?

Traditional models (MLP, CNN on tabular inputs, XGBoost) operate on node features:
They **cannot condition on graph connectivity**.

Graph Neural Networks instead learn via **message passing**, allowing information to propagate across multiple hops. This enables detection of **hierarchical influence patterns** that feature-only models fundamentally miss.

---

## Methodology

### Phase 1 — Synthetic Benchmark

A **controlled synthetic graph** is constructed with three node roles:

- **Meta-Influencers** (influencers of influencers) → positive label  
- **Influencers** → positive label  
- **Regular users** → negative label  

Key design principle:
> Node features intentionally overlap across roles, ensuring that **structure—not features—carries the label signal**.

This creates a clean stress-test for relational inductive bias.

---

### Phase 2 — Real-World Graphs

The full pipeline is repeated on real datasets:

- **StackOverflow (Machine Learning subset)**
- **WikiVote**

For each dataset:
- Graph construction  
- Feature extraction  
- Model training  
- Metric evaluation  
- CPU vs GPU benchmarking  
- Hop-sweep experiments  

---

## Models Evaluated

### Graph Neural Networks
- **GCN** — Graph Convolutional Network  
- **GraphSAGE** — Inductive neighborhood aggregation  
- **GAT** — Graph Attention Network  

### Baselines
- **XGBoost** — strong tabular baseline  
- **Feature-only neural model** (MLP-style)

All models receive identical node features.  
**Only GNNs receive edge information.**

---

## Evaluation Metrics

The following metrics are reported:

- Accuracy  
- Macro-F1  
- ROC / AUC  
- Confusion matrices  
- Training & inference time  
- Learning curves  

Special emphasis is placed on:
- **False negatives** (missed influencers)
- **Meta-influencer precision / recall**

---

## CPU vs GPU Performance Analysis

We benchmark:

- Training time per epoch  
- Full-graph inference time  

Across:
- GNN architectures  
- Model depth (layers)  
- Neighborhood hop count  

### Key Observations

- GPU speedups **increase with depth and hop count**
- Sparse aggregation and memory bandwidth dominate GNN workloads
- Inference speedups often exceed training speedups
- For small graphs or shallow models, CPU can be competitive due to GPU overhead

---

## Graph Visualizer (Qualitative Validation)

An interactive graph visualizer is included to:

- Inspect neighborhood structure  
- Validate multi-hop influence paths  
- Debug real-world connectivity  

This qualitative validation confirms that detected influencers align with **actual structural patterns**, not spurious feature correlations.

---

## Key Findings

- GNNs **consistently outperform** feature-only baselines when influence is structural
- Feature-only models can still perform well when features correlate strongly with labels
- GPU acceleration benefits grow with:
  - deeper models
  - larger hop neighborhoods
- NVIDIA GPUs excel due to:
  - massive parallelism
  - high memory bandwidth
  - efficient sparse operations

---

## Limitations

- Influence labels are proxy-based and may introduce noise  
- Hyperparameter search is not exhaustive  
- Performance trends may vary across graph types (homophily vs heterophily)

---

## Future Work

- Hybrid pipelines:  
  GNN embeddings distilled into fast feature-only models
- Richer heterogeneous graphs with multiple edge types
- Larger-scale GPU benchmarking and profiling

---

## Documentation & Artifacts

- **Full Research Paper (PDF):**  
  `docs/Research findings_Design__Performance__and_Hardware_Aware_Evaluation_of_Graph_Neural_Networks_for_Hierarchical_Influence_Detection.pdf`

- **Presentation Slides:**  
  `docs/Presentation_ASHIK_GNN.pdf`

---

## Conclusion

This project demonstrates that:

1. **Hierarchical influence detection is fundamentally relational**
2. **Graph Neural Networks provide the correct inductive bias**
3. **NVIDIA GPUs deliver substantial acceleration for sufficiently large GNN workloads**

The synthetic-to-real evaluation strategy ensures both **conceptual rigor** and **practical relevance**.


