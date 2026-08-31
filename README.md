# Crowd-Counting-AVSS-2026  

**Lightweight Crowd Counting via Density Map Estimation**  

**Official PyTorch implementation of our conference paper:**  

**Density Regression Refiner Framework with Pyramidal Feature Fusion Networks for Crowd Counting**  

Shahbaz Ahmad, Prithwijit Guha  

AVSS 2026  

Department of Electronics and Electrical Engineering, Indian Institute of Technology Guwahati, India 

This repository provides the source code for our lightweight crowd-counting framework based on density-map regression. The framework combines a pretrained lightweight backbone, attention-based feature refinement, pyramidal multi-scale feature fusion, and a lightweight density regression head to estimate crowd density maps and corresponding crowd counts.

The implementation includes:

Dataset loading and preprocessing
Ground-truth density-map generation
Geometry-Adaptive Kernel (GAK)
Fixed Gaussian Kernel (FGK)
Lightweight pretrained backbone loading
Attention modules
Pyramidal feature fusion modules
Density-map regression
Training and validation
Model checkpointing
Testing and visualization
MAE and RMSE/MSE evaluation
Parameter-count computation
GFLOPs computation
Model-size measurement
Inference latency measurement
FPS measurement


