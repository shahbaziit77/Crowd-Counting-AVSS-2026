# Crowd-Counting-AVSS-2026  

**Lightweight Crowd Counting via Density Map Estimation**  

**Official PyTorch implementation of our conference paper:**  

**Density Regression Refiner Framework with Pyramidal Feature Fusion Networks for Crowd Counting**  
Shahbaz Ahmad, Prithwijit Guha  
Adavanced Visual and Signal-Based Systems (AVSS) 2026  
Department of Electronics and Electrical Engineering, Indian Institute of Technology Guwahati, India 

This repository provides the source code for our **lightweight crowd-counting framework: Density regression refiner framework** based on density-map regression. The framework combines a **pretrained lightweight backbone, attention-based feature refinement**, **pyramidal multi-scale feature fusion**, and a lightweight **density regression head** to estimate crowd density maps and corresponding crowd counts.
The implementation includes:  
1. Dataset loading and preprocessing
2. Ground-truth (GT) density map generation:<br>
   a. Geometry-Adaptive Kernel (GAK)<br>
   b. Fixed Gaussian Kernel (FGK)<br>
3. Lightweight pretrained backbone loading
4. Attention modules
5. Pyramidal feature fusion modules
6. Density regressor (density regression head)
7. Training and validation
8. Model checkpointing
9. Testing and visualization
10. MAE and MSE evaluation
11. Parameter count computation
12. GFLOPs computation
13. Model size measurement
14. Inference latency measurement
15. FPS measurement

# 1. Overview  
Crowd counting aims **to estimate the number of people/individuals** in crowded scenes. Density-map-based approaches predict a spatial density distribution whose integral/sum corresponds to the estimated crowd count.  

For an input image
$I \in \mathbb{R}^{h\times w\times3}$,  

the proposed density regression refiner framework learns a mapping
$f_\theta\rightarrow\hat{D}$, where $\hat{D}=f_\theta(I)$, $\hat{D} \in \mathbb{R}^{h/8\times w/8}$ is the estimated density map.  

The estimated crowd count is obtained as $\hat{c}=\sum_{x,y}\hat{D}(x,y)$.  

The corresponding ground-truth count is $c=\sum_{x,y}D(x,y)$, where $D$ represents the ground-truth (GT) density map.

The main objective of this project is to achieve a favorable **accuracy–efficiency trade-off**, making the model suitable not only for crowd counting benchmarks but also for resource-constrained and edge-computing environments.

# 2. Network Architecture  
The general processing pipeline:  
Input Image $I$  
$\downarrow$  
Pretrained Lightweight Backbone $\rightarrow$ Multi-level Feature Maps  
$\downarrow$  
Feature Refiner Module (FRM): CBAM  
$\downarrow$  
Pyramidal Feature Fusion Module (PFFM): FPN/PANet/BiFPN  
$\downarrow$  
Density Regressor (DR): Pointwise Convolution  
$\downarrow$  
Estimated Density Map $\hat{D}$  

Estimated Crowd Count $\hat{c} = \sum \hat{D}$ (Sum over Density Map)  
Depending on the experimental configuration, different lightweight backbones, FRM: CBAM, and pyramidal feature fusion modules (PFFMs) can be selected.  
The implementation of the principal architecture is contained in:  
**model/CSRnet.py**  
while reusable neural network modules and building blocks are defined in:  
**model/net.py**  

# 3. Repository Structure
Lightweight-Crowd-Counting/
│
├── data/
│   ├── __init__.py
│   └── CSRNet_dataset.py
│
├── model/
│   ├── CSRnet.py
│   └── net.py
│
├── gt_generation/
│   └── density_map_generation.py
│
├── CSRNet_main_train.py
├── CSRNet_test2.py
│
├── README.md
├── requirements.txt
└── LICENSE


