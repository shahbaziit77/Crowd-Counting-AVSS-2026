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
Crowd counting aims to estimate the number of people in crowded scenes. Density-map-based approaches predict a spatial density distribution whose integral/sum corresponds to the estimated crowd count.


