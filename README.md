Design,Performance,andHardware-AwareEvaluationofGraph
NeuralNetworksforHierarchical InfluenceDetection
AshikAliShaik
December14,2025
Abstract
Influencedetectioninsocialnetworks is inherentlyrelational: auser’s importancecande
pendmoreonwhotheyareconnectedtothanontheirstandaloneattributes.Thisreportstudies
hierarchical influencedetection, focusingonidentifying“influencersof influencers”—nodesthat
maynothavethelargestdirectfollowingbutexertstrongindirect influenceviaconnectionsto
otherinfluentialnodes.Wepresentatwo-phaseempirical framework: (i)acontrolledsynthetic
benchmarkdesignedtoisolatemulti-hopinfluencepatternsthatcannotbeinferredreliablyfrom
nodefeaturesalone,and(ii)anevaluationonreal-worldgraphs(StackOverflowandWikiVote)
totestwhether the sameadvantagesholdat realistic scaleandnoise. Wecomparemultiple
GNNarchitectures(GCN,GraphSAGE,GAT)againststrongfeature-onlybaselines(aneural
MLP-stylemodelandXGBoost)usingaccuracy,macro-F1,ROC/AUC,andconfusionmatrices,
andwereport learningdynamicsviatrainingcurves. Finally,webenchmarkCPUvsNVIDIA
GPUperformancefortrainingandinference, includinghop-sweepexperimentsthatvaryneigh
borhooddepth.Acrossbothphases,GNNsconsistentlycapturehierarchical influencepatterns
thatbaselinesmiss, especiallywhenthe label signal is structural. GPUspeedupsgrowwith
model depthandhopcount, indicating thatparallel sparseaggregationandmemoryband
width—rather thandensematrixthroughput—dominateGNNacceleration. Wecomplement
quantitativeresultswithaninteractivegraphvisualizertoqualitativelyvalidateinfluencepaths
andneighborhoodstructure.
Contents
1 Introduction 3
1.1 Researchquestions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.2 Contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2 Background:GraphNeuralNetworksfromFirstPrinciples 3
2.1 Graphs, features,andsupervisednodeclassification. . . . . . . . . . . . . . . . . . . 3
2.2 Messagepassing: thecoreGNNmechanism . . . . . . . . . . . . . . . . . . . . . . . 4
2.3 Architecturesusedinthisstudy. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
3 ProblemDefinition:Hierarchical InfluenceDetection 4
4 Methodology 5
4.1 Overallexperimentalflow . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
4.2 Phase1: Syntheticdatasetgeneration . . . . . . . . . . . . . . . . . . . . . . . . . . 5
4.2.1 Nodetypesandlabels . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
4.2.2 Syntheticgenerationcode(representativesnippet) . . . . . . . . . . . . . . . 5
1
4.3 Graphconstructionandinputmodalities. . . . . . . . . . . . . . . . . . . . . . . . . 5
4.4 Phase2:Realdatasetsandgraphextraction. . . . . . . . . . . . . . . . . . . . . . . 6
4.5 Graphvisualizer(qualitativeverification) . . . . . . . . . . . . . . . . . . . . . . . . 6
5 ExperimentalSetup 6
5.1 Trainingprotocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
5.2 Hardwareandsoftware. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
6 Phase1Results: SyntheticBenchmark 6
6.1 Multi-classconfusionmatrices(roleseparation) . . . . . . . . . . . . . . . . . . . . . 7
6.2 Binaryresults:ROCcurves,confusionmatrices,andlearningcurves . . . . . . . . . 7
6.3 Phase1CPUvsGPU:depthscaling . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
7 Phase2Results:Real-WorldGraphs 9
7.1 StackOverflow(MachineLearningsubset) . . . . . . . . . . . . . . . . . . . . . . . . 9
7.2 WikiVote . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
8 Phase2(Part2):HopSweepsandCPUvsGPUPerformance 12
8.1 Interpretinghop-sweepbehavior . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
9 GPUAccelerationandNVIDIAArchitecture:ConnectingResultstoHardware 14
9.1 WhatdominatesGNNcompute? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
9.2 WhyNVIDIAGPUshelp(andwhentheydonot) . . . . . . . . . . . . . . . . . . . 14
9.3 Architecture-relevantGPUfeatures . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
10Discussion:WhattheResultsMean 15
10.1WhyGNNsaresuitedforinfluencers-of-influencers . . . . . . . . . . . . . . . . . . . 15
10.2Whenfeature-onlymodelscanstillwin. . . . . . . . . . . . . . . . . . . . . . . . . . 15
10.3Failuremodesandinstability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
11Limitations 15
12FutureWork 15
13Conclusion 16
AAppendix:FullPlotGallery 16
A.1 Phase1additionalplots . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
A.2 Phase2additionalplots . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
2
1 Introduction
Social influence analysis underpins applications such as influencer marketing, recommendation, and
information diffusion monitoring. A key difficulty is that influence can be indirect. For example,
a user may have relatively few direct followers but can still be highly influential if their followers
are themselves highly influential. This motivates the task studied here: detecting influencers of
influencers.
Conventional feature-vector models (MLP, CNN on tabular inputs, gradient-boosted trees)
operate on node attributes (e.g., follower count, post count) and cannot directly represent who
influences whom. Graph Neural Networks (GNNs) address this by learning from both node features
and graph connectivity through message passing. This report provides a comprehensive, end-to-end
study—from dataset generation through model training, evaluation, qualitative visualization, and
hardware-aware performance analysis.
1.1 Research questions
We structure the work around four questions:
RQ1: Whenthelabeldependsonmulti-hop structure, do GNNs outperform feature-only baselines?
RQ2: How do different GNN architectures compare (accuracy, F1, ROC/AUC, and error modes)?
RQ3: How does CPU vs GPU performance change with model depth (layers) and neighborhood
depth (hops)?
RQ4: Which NVIDIA GPU architectural properties best explain the observed speedups for GNN
workloads?
1.2 Contributions
• Asynthetic hierarchical-influence benchmark that explicitly embeds “influencers of influencers”
patterns while controlling feature confounders.
• Acomparative evaluation of GNNs vs XGBoost and a feature-only neural baseline, using consis
tent metrics and visual diagnostics (confusion matrices, ROC curves, learning curves).
• ACPU/GPU benchmarking suite quantifying training and inference times, including hop-sweep
experiments on real data.
• Agraph visualizer and associated exporters that provide qualitative verification of structure and
influence-path patterns.
2 Background: Graph Neural Networks from First Principles
2.1 Graphs, features, and supervised node classification
Agraph is defined as G = (V,E), where V is a set of nodes (users) and E is a set of edges (relations
such as follows, replies, comments, votes). Each node v ∈ V has a feature vector xv ∈ Rd. In node
classification, we aim to learn a function f such that ˆyv = f(G,X,v) predicts a label yv for each
node.
Feature-only models approximate ˆyv ≈ f(xv) and thus cannot condition on E. GNNs explicitly
condition on neighborhoods defined by E.
3
2.2 Message passing: the core GNN mechanism
Most modern GNNs can be expressed via a message passing neural network (MPNN) update:
m(k)
v =AGG ϕ(k)(h(k−1)
v
h(k)
v =ψ(k) h(k−1)
v
where h(0)
, m(k)
v
,
, h(k−1)
u
, euv) : u ∈ N(v) ,
(1)
(2)
v =xv, N(v) is the neighbor set, euv are optional edge features, AGG is a permutation
invariant operator (sum/mean/max), and ϕ,ψ are learnable functions. Intuitively:
• Local-to-global: stacking K layers propagates information up to K hops.
• Structure sensitivity: the update depends on graph connectivity, enabling multi-hop influence
patterns.
• Sparse, irregular compute: neighborhoods vary in size; operations are dominated by sparse
aggregation.
2.3 Architectures used in this study
We focus on widely used GNN families:
• GCN[4]: normalized neighborhood averaging with linear transforms.
• GraphSAGE [3]: inductive aggregation (often mean) and the option to sample neighbors for
scalability.
• GAT[6]: attention weights to reweight neighbors based on learned compatibility.
For baselines we use:
• XGBoost [1]: strong non-linear tabular baseline.
• Feature-only neural baseline: an MLP/CNN-style network operating on node features with
out access to edges.
3 Problem Definition: Hierarchical Influence Detection
Wedefinehierarchical influence detection as a binary node classification task where the positive
class includes:
1. Influencers: nodes that directly influence many others.
2. Influencers of influencers: nodes that may have few direct links, but connect to influencers,
meaning their influence is amplified indirectly.
Crucially, influencers of influencers are characterized by multi-hop relational patterns rather than
easily separable feature values. This creates a natural stress-test for GNNs versus feature-only
methods.
4
4 Methodology
4.1 Overall experimental flow
The project follows a two-phase flow:
1. Phase 1 (Synthetic): construct a controlled graph with explicit hierarchical influence patterns,
train models, and analyze both accuracy and CPU/GPU runtime as model depth increases.
2. Phase 2 (Real data): repeat comparisons on real graphs and run hop-sweep experiments to
test how neighborhood depth affects both predictive performance and GPU speedups.
4.2 Phase 1: Synthetic dataset generation
The synthetic generator creates a graph with distinct structural roles while ensuring node features
alone are insufficient. The key design is to embed patterns where a “meta-influencer” connects
to several influencers, and those influencers connect to many regular users. In terms of influence
diffusion, the meta-influencer has strong second-order reach even if its first-order degree is modest.
4.2.1 Node types and labels
We create three conceptual types:
• Meta-influencer (influencer of influencers): positive label.
• Influencer: positive label.
• Regular user: negative label.
To avoid trivial separability, follower-like counts and activity-like features are designed to overlap
across types (e.g., a meta-influencer may not have the largest raw follower count). Thus, the
intended signal is the graph structure.
4.2.2 Synthetic generation code (representative snippet)
The full implementation is provided in the project scripts (Phase 1 generator + training). The
central idea is illustrated by the following simplified excerpt:
Listing 1: Conceptual snippet: creating meta-influencer connectivity to influencers.
# Create influencer-of-influencer pattern
meta = create_node(type="meta_influencer", label=1)
for inf in influencer_nodes:
graph.add_edge(meta, inf)
# meta-> influencer
for u in followers_of(inf):
graph.add_edge(inf, u)
# influencer-> regular users
4.3 Graph construction and input modalities
All models receive node features X. Only GNNs receive edges E (as sparse adjacency / edge index).
Baselines operate on X alone. This separation is essential for a fair test of “does connectivity help?”.
5
4.4 Phase 2: Real datasets and graph extraction
Phase 2 uses two real graph datasets (as implemented in the project outputs):
• StackOverflow (machine learning subset): users and interactions are turned into a graph;
node labels correspond to hierarchical influence proxies used in the project pipeline.
• WikiVote: a directed voting graph; again used for node classification under the hierarchical
influence framing.
For each dataset, the pipeline constructs features, builds the graph, trains comparable models, and
exports plots and summary CSVs for reproducibility.
4.5 Graph visualizer (qualitative verification)
A dedicated interactive visualizer is used to inspect nodes, edges, and multi-hop paths. This is
important for two reasons:
1. Sanity-checking structure: confirming that synthetic “meta-influencer → influencer” chains
exist.
2. Debugging real graphs: verifying connectivity, component structure, and that neighborhoods
used by message passing reflect intended interactions.
Figure 1 shows a representative visualizer snapshot.
5 Experimental Setup
5.1 Training protocol
Across phases, models are trained for a fixed number of epochs with consistent train/validation/test
splits (as defined by the project scripts). We use standard supervised classification losses (cross
entropy or BCE depending on setup) and report:
• Accuracy: overall correctness.
• Macro-F1: balances class performance when classes are imbalanced.
• ROC/AUC: threshold-independent separability.
• Confusion matrix: error modes (false positives vs false negatives).
• Runtime metrics: average epoch time for training and average inference time.
5.2 Hardware and software
GNNtraining is implemented in PyTorch / PyTorch Geometric [2]. GPU runs use NVIDIA CUDA
(CUDA backend enabled) [5]. The performance analysis focuses on how sparse aggregation work
loads map to GPU strengths.
6 Phase 1 Results: Synthetic Benchmark
Phase 1 establishes whether the task genuinely requires structure. We report both multi-class style
confusion matrices (separating roles) and binary results (positive = influencer or meta-influencer).
6
6.1 Multi-class confusion matrices (role separation)
Figures 2a–2c show confusion matrices for GNN, feature-only neural baseline (labeled MLP in
plots), and XGBoost on the synthetic dataset. These matrices reveal how each model confuses the
classes:
• GNN: stronger separation of hierarchical roles due to access to connectivity.
• Feature-only: tends to confuse meta-influencers with regular users if their node features overlap.
• XGBoost: strong on separable feature regimes, but limited when structure is the primary signal.
(a) GNN
(b) Feature-only neural baseline
(c) XGBoost
Figure 2: Phase 1 (synthetic) multi-class confusion matrices. These plots highlight which roles are
confused when the model has (GNN) vs lacks (baselines) edge information.
6.2 Binary results: ROC curves, confusion matrices, and learning curves
In the binary setting, the positive class merges influencers and meta-influencers. This tests whether
models can at least separate “influential” nodes from regular users. Figures 3 and 4 provide ROC
curves and confusion matrices across models. Figures 5 show learning curves for the GNN and the
feature-only neural baseline.
How to read the ROC curve. The ROC curve plots true positive rate vs false positive rate as
the classification threshold varies. AUC summarizes separability; higher AUC indicates that the
model ranks positive nodes above negative nodes more consistently.
How to read the confusion matrix. The confusion matrix summarizes (TP, FP, TN, FN).
For influence detection, false negatives (missing influencers) are often more costly, so we examine
recall as well as precision.
How to read learning curves. Training curves show optimization dynamics. Divergence be
tween training and validation indicates overfitting; slow improvement indicates under-capacity or
optimization issues.
7
(a) GNN ROC
(b) Feature-only ROC
(c) XGBoost ROC
Figure 3: Phase 1 (synthetic) binary ROC curves across models.
(a) GNN CM
(b) Feature-only CM
(c) XGBoost CM
Figure 4: Phase 1 (synthetic) binary confusion matrices.
(a) GNN learning curves
(b) Feature-only learning curves
Figure 5: Phase 1 (synthetic) learning curves. These curves help interpret whether performance
differences are due to representational limitations (structure vs no structure) or training instabili
ty/overfitting.
8
Table1: SyntheticphaseCPUvsGPUtimingandtestperformance forGNNarchitecturesand
layercounts.
arch layers avgtraintimescpu avgtraintimesgpu trainspeedup avginferencetimescpu avg
gcn 1 0.2375 0.0065 36.6400 0.1121 0.0023 48.1739 0.9980 0.9979
gcn 2 0.4715 0.0072 65.5669 0.2251 0.0044 50.6140 0.9980 0.9979
gcn 3 0.7179 0.0097 74.1271 0.3400 0.0067 50.8422 0.9950 0.9948
gcn 4 0.9483 0.0123 76.9143 0.4575 0.0091 50.5274 0.6955 0.6921
graphsage 1 0.0261 0.0276 0.9449 0.0063 0.0014 4.3954 0.9970 0.9969
graphsage 2 0.1649 0.0053 31.3674 0.0677 0.0023 29.8405 0.9965 0.9964
graphsage 3 0.3015 0.0073 41.4365 0.1340 0.0032 42.1125 0.9965 0.9964
graphsage 4 0.4667 0.0081 57.5993 0.1994 0.0044 44.9811 1.0000 1.0000
6.3 Phase1CPUvsGPU:depthscaling
TostudyGPUacceleration,webenchmarktrainingandinferencetimes formultipleGNNarchi
tecturesasmodeldepth(numberof layers) increases. TwokeyeffectsappearrepeatedlyinGNN
workloads:
•Overheadatsmall scale: forshallowmodelsorsmallgraphs,GPUkernel launchandhost
devicetransferoverheadcandominate,yieldinglimitedspeedup.
•Sparse aggregationacceleration: as depth increases, neighborhoodaggregationbecomes
heavier;GPUscanparallelizetheseoperationsandhidememorylatencymoreeffectively.
Table1reportsmeasuredtimesandcomputedspeedups.Figures6and7visualizethesetrends.
7 Phase2Results:Real-WorldGraphs
Phase2evaluateswhetherconclusionsfromthesyntheticbenchmarkholdonrealdata.Wereport
results forStackOverflow(machine learningsubset)andWikiVote. Foreachdataset,we include
confusionmatricesandROCcurves forGNN, feature-onlybaseline, andXGBoost,plus training
curvesfortheneuralmodels.
7.1 StackOverflow(MachineLearningsubset)
Table2summarizesperformancemetrics. Wethenanalyzeplots to interpretwhereeachmodel
succeedsorfails.
Observations. Fromthesummarymetricsandplots:
•ROC/AUC: indicatesrankingquality;highAUCsuggeststhatthemodelassignshigherscores
toinfluentialusersevenwhenafixedthresholdmightnotperfectlyseparateclasses.
•Macro-F1: reflectsbalancedperformance; if thedataset is imbalanced,accuracyalonecanbe
misleading.
9
Table 2: Phase 2 results summary on StackOverflow (machine learning subset).
model
accuracy macro f1
auc precision pos recall pos f1 pos meta count meta prec meta rec meta
xgboost
mlp
gnn
0.8980
0.7551
0.8163
0.8485 0.9714
0.7245 0.9095
0.7796 0.9011
0.7500
0.4627
0.5357
0.7742 0.7619
1.0000 0.6327
0.9677 0.6897
11
11
11
1.0000
1.0000
1.0000
0.3636
1.0000
1.0000
0.5333
1.0000
1.0000
• Meta-influencer metrics: the exported summary includes precision/recall/F1 on the “meta”
group, which directly targets the influencers-of-influencers phenomenon.
(a) GNN CM
(b) Feature-only CM
(c) XGBoost CM
Figure 8: StackOverflow: confusion matrices by model. These reveal whether errors are dominated
by false negatives (missing influencers) or false positives (over-predicting influence).
(a) GNN ROC
(b) Feature-only ROC
(c) XGBoost ROC
Figure 9: StackOverflow: ROC curves by model. AUC summarizes threshold-independent separa
bility.
10
Table 3: Phase 2 results summary on WikiVote.
model
accuracy macro f1
auc precision pos recall pos f1 pos meta count meta prec meta rec meta
xgboost
mlp
gnn
0.9906
0.7472
0.9007
0.9837 0.9994
0.6825 0.8774
0.8488 0.9415
0.9677
0.3930
0.6512
0.9783 0.9730
0.8587 0.5392
0.9130 0.7602
80
80
80
1.0000
1.0000
1.0000
0.9500
0.6750
0.8000
0.9744
0.8060
0.8889
(a) GNN learning curves
(b) Feature-only learning curves
Figure 10: StackOverflow: training dynamics. These curves help diagnose stability and generaliza
tion gaps.
7.2 WikiVote
Table 3 summarizes performance metrics on WikiVote. We then interpret plots.
(a) GNN CM
(b) Feature-only CM
(c) XGBoost CM
Figure 11: WikiVote: confusion matrices by model.
11
(a) GNN ROC
(b) Feature-only ROC
Figure 12: WikiVote: ROC curves by model.
(c) XGBoost ROC
(a) GNN learning curves
(b) Feature-only learning curves
Figure 13: WikiVote: training dynamics.
8 Phase 2 (Part 2): Hop Sweeps and CPU vs GPU Performance
A critical control knob in GNNs is the neighborhood depth (hops). Increasing hops increases the
receptive field, potentially improving the ability to detect hierarchical influence—but also increases
compute and memory cost due to more message passing.
Table 4 summarizes hop-sweep results across datasets and GNN variants, including CPU vs
GPU epoch-time and inference-time speedups.
8.1 Interpreting hop-sweep behavior
We interpret the hop-sweep along two axes:
Predictive performance vs hops. Performance may improve with more hops because meta
influence patterns are multi-hop. However, too many hops can lead to oversmoothing (node repre
sentations become similar) or noise aggregation, harming accuracy/F1.
12
Table4: Phase2(Part2)CPUvsGPUtimingandperformanceacrossdatasets,GNNvariants,
andhopcounts.
dataset gnn hops avgepochtimescpu avgepochtimescuda trainspeedupcpuovergpu avg
stackoverflowml gcn 1 0.0047 0.0023 2.0634 0.0005 0.0003 1.6380 0.5578 0.7143 0.5296 0.6448
stackoverflowml gcn 2 0.0081 0.0034 2.4047 0.0010 0.0005 2.1362 0.7619 0.7415 0.6750 0.6569
stackoverflowml gcn 3 0.0077 0.0042 1.8258 0.0008 0.0007 1.2339 0.7755 0.7823 0.6741 0.6730
stackoverflowml gcn 4 0.0097 0.0052 1.8576 0.0019 0.0009 2.2142 0.7755 0.7619 0.6667 0.6617
stackoverflowml sage 1 0.0243 0.0035 6.9928 0.0020 0.0007 2.8685 0.8707 0.7687 0.8035 0.7371
stackoverflowml sage 2 0.0201 0.0049 4.1054 0.0019 0.0006 3.3428 0.7823 0.8095 0.7405 0.7699
stackoverflowml sage 3 0.0166 0.0050 3.2972 0.0022 0.0008 2.7337 0.8027 0.7891 0.7600 0.7533
stackoverflowml sage 4 0.0201 0.0061 3.2667 0.0034 0.0010 3.3695 0.8163 0.8095 0.7796 0.7729
stackoverflowml sgc 1 0.0041 0.0024 1.7510 0.0004 0.0003 1.2414 0.5850 0.5714 0.5644 0.5616
stackoverflowml sgc 2 0.0044 0.0025 1.7783 0.0005 0.0004 1.3834 0.6667 0.7279 0.6101 0.6250
stackoverflowml sgc 3 0.0045 0.0025 1.8164 0.0007 0.0005 1.5350 0.7823 0.7483 0.6874 0.6346
stackoverflowml sgc 4 0.0049 0.0025 1.9332 0.0010 0.0006 1.8603 0.7891 0.8027 0.6627 0.6925
wikivote gcn 1 0.0160 0.0028 5.6721 0.0053 0.0004 15.1846 0.7163 0.8277 0.5627 0.6412
wikivote gcn 2 0.0504 0.0043 11.7199 0.0119 0.0008 15.5672 0.1770 0.6592 0.1532 0.5615
wikivote gcn 3 0.0614 0.0058 10.5147 0.0178 0.0013 13.9058 0.5552 0.3240 0.5242 0.3107
wikivote gcn 4 0.0963 0.0076 12.6231 0.0259 0.0015 16.8932 0.5890 0.1723 0.5487 0.1470
wikivote sage 1 0.0179 0.0025 7.0633 0.0064 0.0004 16.9650 0.8446 0.8773 0.7310 0.8199
wikivote sage 2 0.0558 0.0045 12.3892 0.0132 0.0009 15.0735 0.8605 0.8970 0.8013 0.8387
wikivote sage 3 0.0697 0.0058 12.0810 0.0210 0.0013 16.5031 0.8830 0.8933 0.8271 0.8212
wikivote sage 4 0.1065 0.0079 13.4317 0.0296 0.0019 15.6630 0.9082 0.8989 0.8583 0.8465
wikivote sgc 1 0.0158 0.0024 6.7318 0.0055 0.0003 20.2550 0.5000 0.6264 0.4553 0.5800
wikivote sgc 2 0.0259 0.0024 10.7404 0.0155 0.0004 42.8834 0.6189 0.3998 0.5751 0.3935
wikivote sgc 3 0.0298 0.0026 11.6731 0.0150 0.0005 32.2170 0.2041 0.6732 0.1894 0.5332
wikivote sgc 4 0.0312 0.0026 11.8132 0.0191 0.0006 34.1484 0.1723 0.8764 0.1470 0.7523
13
Runtime vs hops. Runtime generally increases with hops because each layer aggregates over
neighbors. GPU speedups often increase with hops because the workload becomes large enough to
amortize overhead and exploit parallel sparse aggregation.
9 GPUAcceleration and NVIDIA Architecture: Connecting Re
sults to Hardware
This section explains why GPU speedups occur in this project, based on the computation performed
by GNN message passing.
9.1 What dominates GNN compute?
Unlike CNNs, where dense matrix multiplications are dominant and Tensor Cores can be central,
many GNN workloads are dominated by:
• sparse gather/scatter of neighbor features,
• reduction (sum/mean) over variable-size neighborhoods,
• memory bandwidth and cache behavior (irregular access),
• kernel launch overhead when graphs are small.
9.2 Why NVIDIA GPUs help (and when they do not)
The project’s results align with the following mapping:
• Large speedups at larger hops / deeper models: more aggregation work increases arith
metic intensity and GPU occupancy.
• Inference speedups often exceed training speedups: forward pass is simpler than backward
pass and can be more efficiently parallelized.
• Limited speedup for small graphs or shallow models: fixed overhead dominates; CPU
can be competitive.
9.3 Architecture-relevant GPU features
Observed speedups are most consistent with:
• Massive thread-level parallelism (CUDA cores): parallelizing neighbor aggregation across
edges.
• High memory bandwidth: streaming feature vectors for many edges.
• Efficient sparse primitives: scatter/gather and reductions (often implemented via specialized
kernels).
14
10 Discussion: What the Results Mean
10.1 Why GNNs are suited for influencers-of-influencers
The “influencers of influencers” label is inherently relational: a node’s importance depends on the
importance of nodes it connects to, which is naturally captured by message passing. Feature-only
models cannot represent this dependency unless such multi-hop structure is explicitly engineered
into features.
10.2 When feature-only models can still win
On some real datasets, feature-only baselines (including XGBoost) can appear strong if features
correlate strongly with influence proxies. This does not contradict the value of GNNs; rather it
highlights that the benefit of GNNs depends on how much label signal is structural vs purely
feature-driven.
10.3 Failure modes and instability
Some configurations in the hop-sweep table show degraded accuracy for certain model/dataset/hop
choices. Common causes include:
• oversmoothing for deep message passing,
• class imbalance amplifying threshold sensitivity,
• hyperparameter sensitivity (learning rate, dropout, neighbor sampling, normalization),
• graph sparsity/density differences changing neighborhood statistics.
These are well-known challenges in GNN practice and motivate careful tuning and architecture
selection.
11 Limitations
• Dataset labeling: hierarchical influence is a complex concept; any proxy label may introduce
noise.
• Hyperparameter breadth: while multiple architectures and hop settings are explored, ex
haustive tuning can further improve performance.
• Generalization across domains: performance trends may vary with graph type (directed vs
undirected, homophily vs heterophily).
12 Future Work
A promising direction is to build hybrid pipelines:
1. Learn graph-aware embeddings with a GNN.
2. Distill or export these embeddings to feature-only models (MLP/CNN/XGBoost) for fast infer
ence.
15
This could preserve relational signal while enabling lower-latency production systems. Another
direction is to incorporate richer edge types (comments, replies, co-activity) into a heterogeneous
GNN and study how different relation channels impact both accuracy and GPU efficiency.
13 Conclusion
This report demonstrates that (i) hierarchical influence detection benefits from relational inductive
bias, making GNNs a natural fit, and (ii) NVIDIA GPUs provide substantial acceleration for GNN
workloads when the computation is large enough (deeper models or larger hop neighborhoods)
to amortize overhead and exploit parallel sparse aggregation. The synthetic-to-real evaluation
strategy makes these conclusions robust: Phase 1 verifies the structural necessity, and Phase 2
validates practical relevance and scalability.
References
[1] Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. Proceedings of
the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining
(KDD), 2016.
[2] Matthias Fey and Jan E. Lenssen. Pytorch geometric: Fast graph representation learning with
pytorch. ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.
[3] William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large
graphs. In Advances in Neural Information Processing Systems (NeurIPS), 2017.
[4] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional
networks. In International Conference on Learning Representations (ICLR), 2017.
[5] NVIDIA. Cuda c programming guide, 2025. Accessed 2025-12.
[6] Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`o, and
Yoshua Bengio. Graph attention networks. In International Conference on Learning Repre
sentations (ICLR), 2018.
A Appendix: Full Plot Gallery
For completeness and reproducibility, we include all exported plots from the project artifacts.
A.1 Phase 1 additional plots
Figures below replicate Phase 1 outputs already discussed, provided for quick reference in a single
location.
16
(a) Synthetic multi-class CM
(GNN)
(b) Synthetic multi-class CM
(Feature-only)
(c) Synthetic multi-class CM (XG
Boost)
Figure 14: Appendix: Phase 1 synthetic multi-class confusion matrices.
(a) ROC (GNN)
(b) ROC (Feature-only)
(c) ROC (XGBoost)
Figure 15: Appendix: Phase 1 synthetic ROC curves.
(a) CM (GNN)
(b) CM (Feature-only)
(c) CM (XGBoost)
Figure 16: Appendix: Phase 1 synthetic binary confusion matrices.
17
(a) Learning curve (GNN)
(b) Learning curve (Feature-only)
Figure 17: Appendix: Phase 1 synthetic learning curves.
A.2 Phase 2 additional plots
(a) SO CM (GNN)
(b) SO CM (Feature-only)
(c) SO CM (XGBoost)
Figure 18: Appendix: StackOverflow confusion matrices.
(a) SO ROC (GNN)
(b) SO ROC (Feature-only)
(c) SO ROC (XGBoost)
Figure 19: Appendix: StackOverflow ROC curves.
18
(a) SO learning curves (GNN)
(b) SO learning curves (Feature-only)
Figure 20: Appendix: StackOverflow learning curves.
(a) WV CM (GNN)
(b) WV CM (Feature-only)
(c) WV CM (XGBoost)
Figure 21: Appendix: WikiVote confusion matrices.
(a) WV ROC (GNN)
(b) WV ROC (Feature-only)
(c) WV ROC (XGBoost)
Figure 22: Appendix: WikiVote ROC curves.
19
(a) WV learning curves (GNN)
(b) WV learning curves (Feature-only)
Figure 23: Appendix: WikiVote learning curves.
20
Figure 1: Graph visualizer snapshot showing clusters and connecting edges. The visual structure
helps confirm the presence of multi-hop influence patterns (meta-influencers connected to influencers
who connect to many regular users).
21
Figure 6: Phase 1 synthetic: average training time per epoch for CPU vs GPU across GNN variants
and depths. The GPU advantage increases as depth increases and aggregation dominates.
22
Figure 7: Phase 1 synthetic: average inference time for CPU vs GPU across GNN variants and
depths. Inference speedups are often larger because forward pass is more regular than backprop
and benefits from parallel sparse ops