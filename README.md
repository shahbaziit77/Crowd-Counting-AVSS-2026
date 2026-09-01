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

The main objective of this project is to achieve a favourable **accuracy–efficiency trade-off**, making the model suitable not only for crowd counting benchmarks but also for resource-constrained and edge-computing environments.

# 2. Network Architecture  
The general network architecture pipeline:  
Input Image $I$  
$\downarrow$  
Pretrained Lightweight Backbone $\rightarrow$ Multi-scale Feature Extraction  
$\downarrow$  
Feature Refiner Module (FRM): CBAM  
$\downarrow$  
Pyramidal Feature Fusion Module (PFFM): FPN/PANet/BiFPN $\rightarrow$ Multi-scale Feature Fusion    
$\downarrow$  
Density Regressor (DR): Pointwise Convolution ($Conv 1\times1 $)    
$\downarrow$  
Estimated Density Map $\hat{D}$  

Estimated Crowd Count $\hat{c} = \sum \hat{D}$ (Sum over Density Map)  
Depending on the ablation experimental configuration, different **lightweight backbones: MobileNetV1/V2/V3, ShuffleNetV2, EfficientNet-B0, BBLiteV4**, **FRM: Convolutional block attention module (CBAM)**, and **pyramidal feature fusion modules (PFFMs): FPN, PANet, BiFPN**, can be selected.  

# 3. Repository Structure
```text
Lightweight Crowd Counting: Density Regression Refiner Framework/
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
├── BBLiteV4.py
|
├── CSRNet_main_train.py
├── CSRNet_test2.py
│
├── README.md
├── requirements.txt
└── LICENSE
```
### 📂 File Descriptions

- **`data/`** — Dataset loading and preprocessing.
  - **`CSRNet_dataset.py`** — This script includes:<br>
         1. Dataset loading: Reading crowd images and the corresponding ground-truth (GT) density maps.<br>
         2. Dataset path generation: generating image and GT paths.<br>
         3. Train/validation/test data handling.<br>
         4. Image preprocessing.<br>
         5. Image resizing.<br>
         6. Density map resizing and count-preserving scaling.<br>
         7. Data augmentation.<br>
         8. PyTorch Dataset implementation.<br>
         9. Returning image/density-map pairs to the DataLoader.<br>
  - **`__init__.py`** — Initializes the `data` directory as a Python package and enables importing dataset-related modules.

- **`model/`** — Network architectures and model components.
  - **`CSRnet.py`** — Main crowd counting architecture file includes or integrates:<br>
         1. Pretrained lightweight backbones (multi-scale feature extraction).<br>
         2. Feature refiner module (FRM): CBAM (attention-based feature refinement).<br>
         3. Pyramidal feature fusion modules (PFFMs): FPN/PANet/BiFPN (multi-scale/pyramidal feature fusion).<br>
         4. Density regression head (density regressor (DR)).<br>
         5. Final density map estimation.
  - **`net.py`** — Contains reusable network components and supporting modules required by the main architecture.<br>
         Depending on the selected configuration, this file may contain implementations of:<br>
         1. Convolutional blocks.<br>
         2. Depthwise convolutions.<br>
         3. Pointwise convolutions.<br>
         4. Attention module: CBAM (channel attention, spatial attention).<br>
         5. Feature pyramid modules: FPN, PANet, BiFPN.<br>
         6. Upsampling/downsampling layers.<br>
         7. Density regression modules.

- **`gt_generation/`** — Ground truth (GT) density map generation.
  - **`density_map_generation.py`** — Generates ground truth (GT) crowd density maps from annotated head locations.<br>
         Two density-map generation strategies are supported:<br>
         1. Geometry-adaptive kernel (GAK).<br>
         2. Fixed Gaussian kernel (FGK).<br>
         The generated density maps are used as regression targets during network training.
 
- **`BBLiteV4.py`** — This script includes the pretrained BBLiteV4 lightweight backbone. 

- **`CSRNet_main_train.py`** — Main training and validation script.<br>
       The script contains the complete training pipeline, including:<br>
       1. Model initialization.<br>
       2. Dataset loading.<br>
       3. Training.<br>
       4. Validation.<br>
       5. Loss computation.<br>
       6. Optimizer configuration.<br>
       7. Learning rate (LR) configuration and scheduling.<br>
       8. Model checkpoint saving.<br>
       9. Best model selection.<br>
       10. Parameter count: Params (M) calculation.<br>
       11. GFLOPs calculation.<br>
       12. Model size calculation.<br>
       13. Density map quality: PSNR and SSIM calculation.<br>
       14. Inference latency and FPS measurement.<br>
       15. Training/validation performance monitoring.

- **`CSRNet_test2.py`** — Main testing and visualization script. It performs:<br>
       1. Trained checkpoint loading.<br>
       2. Test set inference.<br>
       3. Crowd count estimation.<br>
       4. MAE/MSE computation.<br>
       5. Input image, ground truth (GT) density map, and estimated density map visualization.<br>
       6. Ground truth (GT) count vs. estimated count comparison.

- **`requirements.txt`** — Python package dependencies.

- **`README.md`** — Repository documentation.

- **`LICENSE`** — Repository license.

## 4. Count Preservation  
An important property of the generated density map is  
$\sum_{x,y}D(x,y)\approx N$, where $N$ is the number of annotated people (actual crowd count).  
Therefore, integrating/summing the density map produces the corresponding crowd count $c = \sum_{x,y}D(x,y)$.  
When resizing a density map, its values should be adjusted so that its total density remains approximately invariant.  
For resizing: $h\times w \rightarrow h'\times w'$,  
a count-preserving scaling can be written as $D'=Resize(D)\frac{hw}{h'w'}$.  
Thus, $\sum D'\approx\sum D$.  

## 5. Datasets  
The framework can be adapted to commonly used crowd counting benchmark datasets such as:<br>  
1. ShanghaiTech: ShanghaiTech Part_A (SHT_A), ShanghaiTech Part_B (SHT_B).<br>
2. UCF_CC_50.<br>
3. UCF-QNRF.

Other point-annotation-based crowd-counting datasets.  
The exact dataset organization should be modified according to the path generation logic in:  
data/CSRNet_dataset.py

A typical dataset structure is
```text
dataset/
│
├── train/
│   ├── images/
│   └── ground_truth/
│
├── val/
│   ├── images/
│   └── ground_truth/
│
└── test/
    ├── images/
    └── ground_truth/
```

## 8. Dataset Preparation  
**Step 1: Download the dataset**<br>
Download the required crowd counting benchmark dataset from its official source.<br>
For example:<br>
1. ShanghaiTech.<br>
2. UCF_CC_50.<br>
3. UCF-QNRF.

Please follow the licenses and terms of use provided by the respective dataset owners.

**Step 2: Organize images and annotations**<br>
Place images and corresponding annotation files in the appropriate directories.<br>
For example:<br>
```text
ShanghaiTech/
│
├── part_A_final/
│   ├── train_data/
│   │   ├── images/
│   │   └── ground-truth/
│   │
│   └── test_data/
│       ├── images/
│       └── ground-truth/
│
└── part_B_final/
    ├── train_data/
    |   ├── images/
    │   └── ground-truth/
    |
    └── test_data/
        ├── images/
        └── ground-truth/
```

**Step 3: Configure dataset paths**<br>
Update the required paths in:<br>
data/CSRNet_dataset.py  
according to the local dataset location.  

## 9. Generate Ground Truth Density Maps  
Ground truth (GT) density maps should be generated before model training.  
Run:<br>
python gt_generation/density_map_generation.py  
Select/configure the desired GT density generation method inside the script.  

**Geometry-Adaptive Kernel (GAK)**  
Annotations  
$\downarrow$  
k-Nearest Neighbors  
$\downarrow$  
Local Distance Estimation
$\downarrow$   
Adaptive $\sigma_i$
$\downarrow$   
Gaussian Kernels    
    
GT Density Map
Fixed Gaussian Kernel
Annotations
     │
     ▼
Fixed σ
     │
     ▼
Gaussian Kernels
     │
     ▼
GT Density Map

Verify that the generated maps approximately satisfy

[
\sum D \approx N.
]



