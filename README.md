# Crowd-Counting-AVSS-2026  

**Lightweight Crowd Counting via Density Map Estimation**  

**Official PyTorch implementation of our conference paper:**  

**Density Regression Refiner Framework with Pyramidal Feature Fusion Networks for Crowd Counting**  

Shahbaz Ahmad, Prithwijit Guha  

Adavanced Visual and Signal-Based Systems (AVSS) 2026  

Department of Electronics and Electrical Engineering, Indian Institute of Technology Guwahati, India 

This repository provides the source code for our lightweight crowd-counting framework: Density regression refiner framework based on density-map regression. The framework combines a pretrained lightweight backbone, attention-based feature refinement, pyramidal multi-scale feature fusion, and a lightweight density regression head to estimate crowd density maps and corresponding crowd counts.

The implementation includes:

1. Dataset loading and preprocessing
2. Ground-truth (GT) density map generation:<br>
   a. Geometry-Adaptive Kernel (GAK)<br>
   b. Fixed Gaussian Kernel (FGK)<br>
4) Lightweight pretrained backbone loading
5) Attention modules
6) Pyramidal feature fusion modules
7) Density map regressor
8) Training and validation
9) Model checkpointing
10) Testing and visualization
11) MAE and MSE evaluation
12) Parameter count computation
13) GFLOPs computation
14) Model size measurement
15) Inference latency measurement
16) FPS measurement


