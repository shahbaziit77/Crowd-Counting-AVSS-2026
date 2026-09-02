# Crowd-Counting-AVSS-2026  

**Lightweight Crowd Counting via Density Map Estimation**  

**Official PyTorch implementation of our conference paper:**  

**A Density Regression Refiner Framework with Pyramidal Feature Fusion Networks for Crowd Counting**  
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
**Download Datasets**
| Datset | Official Download | 
|---|---|
| ShanghaiTech Part_A & Part_B | https://github.com/desenzhou/ShanghaiTechDataset  | 
| UCF_CC_50 | https://www.crcv.ucf.edu/data/ucf-cc-50/ |
| UCF-QNRF | https://www.crcv.ucf.edu/data/ucf-qnrf/ |  

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
$\downarrow$  
GT Density Map  

**Fixed Gaussian Kernel (FGK)**  
Annotations  
$\downarrow$  
Fixed $\sigma$  
$\downarrow$  
Gaussian Kernels  
$\downarrow$  
GT Density Map  

Verify that the generated maps approximately satisfy: $\sum D \approx N$.  

## 10. Training  
Run the main training script:<br>
python CSRNet_main_train.py  

The complete training pipeline:  
Training Images + GT Density Maps  
$\downarrow$  
Dataset  
$\downarrow$  
DataLoader  
$\downarrow$  
Lightweight Crowd Counting (CC) Model: Density Regression Refiner Framework  
$\downarrow$  
Estimated Density Map  
$\downarrow$  
Loss Function  
$\downarrow$  
Backpropagation  
$\downarrow$  
Optimizer  
$\downarrow$  
LR Scheduler  
$\downarrow$  
Parameter Update  
$\downarrow$  
Validation  
$\downarrow$  
Best Checkpoint Saving  

## 11. Model Checkpoints  
During training, model checkpoints are saved by:<br>
CSRNet_main_train.py  

A typical checkpoint directory may be
```text
weights/EfficientNet_B0_CBAM_BiFPN/training01_256x256_val_256x256/saved_model/ (for SHT_A dataset)
│
├── model_checkpoint.pth
├── model_MAE_<best_MAE_epoch_number>.pth (for e.g., model_MAE_114.pth)
└── ...
```

The best checkpoint can be selected using validation performance.  
Example criterion:  
Lowest validation MAE  
or another criterion specified in the training script.  

## Pretrained and Trained Weights  
The pretrained backbone weights and the trained checkpoint of our proposed lightweight crowd counting (CC) model: density regression refiner framework, are provided through Google Drive.

**Download Weights**
| Weight | Model | Download Path |
|---|---|---|
| mobilenetV1X0.25_pretrain.tar | Pretrained MobileNetV1_0.25 backbone weights | Google Drive [https://drive.google.com/drive/u/0/folders/1408cv4cNRZVrJEgKEc4EpIkAV68f5euk] |
| MobileNetV1x0_5.tar | Pretrained MobileNetV1_0.5 backbone weights | Google Drive [https://drive.google.com/drive/u/0/folders/1408cv4cNRZVrJEgKEc4EpIkAV68f5euk] |
| MobileNetV1.tar | Pretrained MobileNetV1_1.0 backbone weights | Google Drive [https://drive.google.com/drive/u/0/folders/1408cv4cNRZVrJEgKEc4EpIkAV68f5euk] |
| BBLiteV4.pth.tar | Pretrained BBLiteV4 backbone weights | Google Drive [https://drive.google.com/drive/u/0/folders/1408cv4cNRZVrJEgKEc4EpIkAV68f5euk] |
| model_checkpoint.pth | Trained EfficientNet-B0 + CBAM + BiFPN crowd counting model checkpoint | Google Drive [https://drive.google.com/drive/u/0/folders/10rMD-6GSbI5X3kCADl0nggOaT-5C_r-U] |  

Note: Replace the Google Drive placeholders above with the public/shareable links to the corresponding files.  

**Directory Structure**  
After downloading the weights, place them in the following directories:
```text
weights/
│
├── Pretrained_weights/
│   └── mobilenetV1X0.25_pretrain.tar
|   └── MobileNetV1x0_5.tar
|   └── MobileNetV1.tar
|   └── BBLiteV4.pth.tar
│
├── EfficientNet_B0_CBAM_BiFPN/        (For SHT_A dataset)
|    └── training01_256x256_val_256x256/
|        └── saved_model/
|            └── model_checkpoint.pth
|            └── model_MAE_<best_MAE_epoch_number>.pth (for e.g., model_MAE_114.pth)
|
└── EfficientNet_B0_CBAM_BiFPN_UCF-QNRF/     (For UCF-QNRF dataset)      
    └── training01_256x256_val_256x256/
        └── saved_model/
            └── model_checkpoint.pth
            └── model_MAE_<best_MAE_epoch_number>.pth (for e.g., model_MAE_67.pth)
```

The complete relevant repository structure should therefore look like:
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
├── weights/
│   ├── Pretrained_weights/
│   │   └── MobileNetV1.tar
│   │
│   └── EfficientNet_B0_CBAM_BiFPN/
│       └── training01_256x256_val_256x256/
│           └── saved_model/
│               └── model_checkpoint.pth
│
├── CSRNet_main_train.py
├── CSRNet_test2.py
├── requirements.txt
├── LICENSE
└── README.md
```

The checkpoint can then be loaded for evaluation using the testing script:<br>
python CSRNet_test2.py  
Weight Usage  

The overall workflow is:<br>
```text
MobileNetV1.tar
      │
      ▼
Pretrained Backbone Initialization
      │
      ▼
Crowd-Counting Network
      │
      ▼
Training
      │
      ▼
model_checkpoint.pth
      │
      ▼
Checkpoint Loading
      │
      ▼
CSRNet_test2.py
      │
      ▼
Test Image
      │
      ▼
Estimated Density Map
      │
      ▼
Estimated Crowd Count
```

**Important**  
The weight files are hosted externally because trained model checkpoints can be relatively large. They are therefore not necessarily included directly in this GitHub repository.  
Please ensure that the downloaded files retain their original filenames and are placed in the exact directory structure shown above before running the corresponding training or testing scripts.

## 12. Testing  
After training, evaluate the model using:<br>
python CSRNet_test2.py  

Before running the script, configure:<br>
1. Test dataset path.<br>
2. Ground truth (GT) density map path.<br>
3. Model configuration.<br>
4. Checkpoint path.<br>
5. Output/visualization path.<br>

The testing pipeline:  
Test Image  
$\downarrow$  
Load Trained Checkpoint  
$\downarrow$  
Forward Propagation  
$\downarrow$  
Estimated Density Map: Density Visualization  
$\downarrow$  
Sum Density Values  
$\downarrow$  
Estimated Count  
$\downarrow$  
Compare with GT Count  
$\downarrow$  
MAE / MSE  

## 13. Complexity and Efficiency Evaluation  
In addition to counting accuracy, the framework evaluates computational efficiency.  
The following metrics are supported in:<br>
CSRNet_main_train.py  

**Parameters**  
The total number of learnable parameters is reported in millions:
$Params(M)=\frac{\text{Number of Parameters}}{10^6}$.  

**GFLOPs**  
The computational cost is reported using **giga floating-point operations**:  
$GFLOPs=\frac{FLOPs}{10^9}$.  
Because GFLOPs depend on the input resolution, always report the resolution used for profiling.  
Example:  
Input resolution for complexity analysis: $768\times1024$.  

**Model Size**  
The trained model size is reported in megabytes: Model Size (MB).  
This provides an additional indication of memory/storage requirements.  

**Inference Latency**  
Latency measures the time required for one forward inference:
$Latency = t_{end}-t_{start}$.  
It is reported in milliseconds:  
Latency (ms/image)  
For GPU evaluation, synchronization should be performed before and after timing to obtain reliable measurements.  

**Frames Per Second (FPS)**  
Inference throughput can be calculated as
$FPS=\frac{1000}{\text{Latency in milliseconds}}$.  
Higher FPS indicates faster inference.  

For fair comparisons, the hardware, input resolution, batch size, precision, warm-up procedure, and timing protocol should be reported together with FPS/latency.

## 14. Accuracy–Efficiency Trade-off  
The lightweight architecture is designed to balance:
```text

             Counting Accuracy
                    ▲
                    │
                    │
Low Complexity ◄────┼────► High Complexity
                    │
                    │
                    ▼
             Deployment Cost
``` 
             
The principal design objective is to obtain competitive counting accuracy while reducing:<br>  
1. Number of parameters: Params (M)<br>  
2. GFLOPs<br>
3. Model size<br>
4. Inference latency<br>

and improving:<br>  
1. FPS<br>
2. Memory efficiency<br>
3. Deployment feasibility<br>

## 15. Visualization  
CSRNet_test2.py can be used to visualize the qualitative crowd-counting results.  

A typical output contains:
```text
┌─────────────────┬──────────────────┬─────────────────────┐
│   Input Image   │  GT Density Map  │ Estimated Density   │
│                 │                  │        Map          │
├─────────────────┼──────────────────┼─────────────────────┤
│   Crowd Scene   │   GT Count: C    | Estimated Count: Ĉ  |
└─────────────────┴──────────────────┴─────────────────────┘
``` 

These visualizations help analyze:<br>
1. Localization of dense crowd regions.<br>   
2. Sparse vs. dense scenes.<br>
3. Perspective variations.<br>
4. Density estimation quality.<br>
5. Ground truth vs. estimated crowd counts.<br>

## 16. Reproducibility Workflow  
The recommended end-to-end workflow:  
STEP 1: Download Crowd-Counting Dataset  
$\downarrow$  
STEP 2: Prepare Images + Point Annotations  
$\downarrow$  
STEP 3: Generate GT Density Maps<br> 
density_map_generation.py  
$\downarrow$  
STEP 4: Configure Dataset Paths<br>  
data/CSRNet_dataset.py  
$\downarrow$  
STEP 5: Configure Network<br>
model/CSRnet.py<br>
model/net.py  
$\downarrow$  
STEP 6: Train + Validate<br>
CSRNet_main_train.py  
$\downarrow$       
STEP 7: Save Best Checkpoint  
$\downarrow$        
STEP 8: Test<br>
CSRNet_test2.py  
$\downarrow$  
STEP 9: Compute MAE / MSE  
$\downarrow$  
STEP 10: Visualize Density Maps  
$\downarrow$  
STEP 11: Report Params / GFLOPs / Model Size / Latency / FPS  

## 17. Example Commands
Generate density maps<br>
python gt_generation/density_map_generation.py  

Train<br>
python CSRNet_main_train.py  

Test<br>
python CSRNet_test2.py  

## 18. Environment  
The implementation is based on Python and PyTorch.  
Recommended environment:<br>
Python >= 3.8<br>
PyTorch<br>
torchvision<br>
NumPy<br>
SciPy<br>
OpenCV<br>
Pandas<br>
Matplotlib<br>
h5py<br>
tqdm<br>
Pillow  

Additional packages may be required for model complexity computation.  
Examples include:<br>
ptflops<br>
thop  
depending on the implementation used in CSRNet_main_train.py.

## 19. Installation  
Clone the repository:<br>
git clone https://github.com/shahbaziit77/Crowd-Counting-AVSS-2026.git<br>
cd Crowd-Counting-AVSS-2026  

Create a virtual environment if desired:<br>
python -m venv crowd_counting_env<br>
source crowd_counting_env/bin/activate  

Install PyTorch according to your CUDA configuration, and then install the remaining dependencies:<br>
pip install -r requirements.txt  

Alternatively, install the principal dependencies manually:<br>
pip install numpy scipy pandas matplotlib opencv-python h5py tqdm pillow  

## 20. Requirements  
A typical requirements.txt may contain:<br>
torch<br>
torchvision<br>
numpy<br>
scipy<br>
opencv-python<br>
pandas<br>
matplotlib<br>
h5py<br>
Pillow<br>
tqdm<br>
ptflops  

Please modify package versions according to the environment used to reproduce the experiments.  
For maximum reproducibility, version-pinned dependencies are recommended.

## 21. Hardware  
Experiments were performed using:<br>
GPU       : NVIDIA A100 80GB PCIe GPU<br>
CPU       : <CPU model, if relevant><br>
CUDA      : CUDA version: 13.2<br>
PyTorch   : PyTorch 13.2<br>
Python    : Python 3.10.6<br>
OS        : Ubuntu 22.04 LTS  

Inference latency and FPS depend strongly on the hardware and software environment. Therefore, efficiency results should be compared under equivalent conditions whenever possible.  

## 22. Notes on Reproducibility  
Results may vary slightly due to:<br>
Random initialization<br>
Dataset preprocessing<br>
Random cropping (optional)<br>
Data augmentation<br>
CUDA/cuDNN implementation<br>
GPU architecture<br>
PyTorch version<br>
Density map generation parameters<br>
Numerical precision<br>
Random seed  

For improved reproducibility, explicitly fix random seeds for Python, NumPy, and PyTorch where applicable.  

## 23. Citation  
If you use this repository, model, or implementation in your research, please cite our paper:<br>
@inproceedings{ahmad2026lightweight,<br>
  title     = {A Density Regression Refiner Framework with Pyramidal Feature Fusion Networks for Crowd Counting},<br>
  author    = {Ahmad, Shahbaz and Guha, Prithwijit},<br>
  booktitle = {22nd International Conference on Adavanced Visual and Signal-Based Systems (AVSS)},<br>
  year      = {2026},<br>
  pages     = {<Pages>},<br>
  doi       = {<DOI>}<br>
}

Please replace this entry with the official bibliographic information after publication.

## 24. Paper

Paper:
<Official Paper/DOI Link>

Conference:
<Conference Website>

Code:
<GitHub Repository Link>

## 25. License  
This repository is released under the <LICENSE NAME> License.

Please see

LICENSE

for additional information.

The external datasets and pretrained backbone weights used by this project remain subject to their respective licenses and terms of use.

## 25. Acknowledgements  
We acknowledge the authors and maintainers of the crowd counting datasets, pretrained backbone architectures, and open-source libraries used in this research.  
This implementation is developed using PyTorch and related scientific Python libraries.  
If code from an external repository has been adapted, its original authors and license should also be acknowledged here.

## 26. Contact  
For questions regarding the paper or implementation, please contact:<br>
Shahbaz Ahmad<br>
Department of Electronics and Electrical Engineering<br>
Indian Institute of Technology Guwahati (IIT Guwahati)<br>
Guwahati, Assam, India  

Email: a.shahbaz@iitg.ac.in

GitHub: https://github.com/shahbaziit77  

## 27. Repository Summary  
```text
Lightweight Crowd Counting
│
├── Data Preparation
│   └── CSRNet_dataset.py
│
├── Ground Truth Generation
│   ├── Geometry-Adaptive Kernel (GAK)
│   └── Fixed Gaussian Kernel (FGK)
│
├── Model
│   ├── Lightweight Backbone
│   ├── Attention Module
│   ├── Pyramidal Feature Fusion
│   └── Density Regression
│
├── Training
│   ├── Train
│   ├── Validation
│   ├── Loss
│   ├── Optimizer
│   ├── LR Scheduler
│   └── Checkpoint Saving
│
├── Complexity
│   ├── Parameters (M)
│   ├── GFLOPs
│   ├── Model Size (MB)
│   ├── Latency (ms)
│   └── FPS
│
└── Testing
    ├── Checkpoint Loading
    ├── Crowd Count Estimation
    ├── MAE / MSE
    ├── GT Density Map
    └── Estimated Density Map
```

## Disclaimer
This repository is intended primarily for academic research and reproducibility of the associated conference paper. Dataset licenses, pretrained model licenses, and third-party software licenses should be respected separately.


