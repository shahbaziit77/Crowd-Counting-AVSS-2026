# Importing the libraries
import os
import glob
import h5py 
import sys
import warnings   
from PIL import Image
from PIL import ImageFilter, ImageDraw
from PIL import ImageStat

import shutil
import math 
import numpy as np                                                          
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from image import *     
import random                          
from tqdm import tqdm  
# from tqdm.notebook import tqdm                                                                                        
import dataset                     
import time
from datetime import datetime                 
import argparse
import copy 

import torch 
import torch.nn as nn 
import torch.nn.functional as F                                                                                    
from torch.utils.data import Dataset, DataLoader 
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn   
import torchvision
from torchvision import datasets, transforms 
# import torchvision.transforms.functional as TF 
# from torchvision.transforms import ToTensor 

from data.CSRNet_dataset import CrowdCountingDataset_v3, CrowdCountingDataset_v3_test, data_path_gen_SHT
from data.CSRNet_dataset import CrowdCountingDataset_v3, CrowdCountingDataset_v3_test, data_path_gen_UCF_QNRF

from model.CSRnet import CoarseNet
from model.CSRnet import CascadedCrowdNet

# %matplotlib inline 

from ptflops import get_model_complexity_info as cps   

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   
# Root of the ShanghaiTech/UCF-QNRF dataset
root_SHT = '/home/pguha5/Shahbaz/ShanghaiTech_dataset_New2'      # ShanghaiTech
root_UCF_QNRF = '/home/pguha5/Shahbaz/UCF-QNRF_dataset_New2'     # UCF-QNRF

data_path = os.path.join(root_SHT, 'part_A_final')     # ShanghaiTech Part_A (SHT_A)
# data_path = root_UCF_QNRF                              # UCF-QNRF

#---------------------------
# Device configuration
#---------------------------
cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print(device) 

#----------------------------
# Hyperparameters
#----------------------------
batch_size = 8
batch_size_test = 1  
num_workers = 4
initial_lr = 1e-4   # Initial LR   
weight_decay = 5*1e-4   

# momentum = 0.95
# factor = 0.5
# patience = 5
# step_size = 20   
# gamma = 0.1
# T_max = 50

lambda_mse = 1.0
lambda_count = 0.1
lambda_ssim = 0.01   
lambda_dm = 1.0

epochs = 400
warmup_ratio = 0.05      # 5% warmup (use 0.10 for 10%)      
warmup_epochs = int(epochs * warmup_ratio)  

print_freq = 10
early_stop_patience = 30
early_stop_counter = 0  

best_loss = float('inf')
best_mae = float('inf') 
best_mse = float('inf') 

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)    


# Initialize model with Xavier initialization 
# Define the model
model = CoarseNet()  # crowd density estimator
model = model.cuda()


print("Using device:", device)
print("Process PID:", os.getpid())
if device.type == 'cuda':
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Memory Allocated:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
    print("Memory Reserved:", torch.cuda.memory_reserved(0) / 1024**2, "MB")

total, used, free = shutil.disk_usage("/")
print("Free space (GB):", free // (1024**3))   

# Params (M) and Model Size (MB) Computations
print(f"================ Params (M) and Model Size (MB) Computations ========================\t")    

params = sum(p.numel() for p in model.parameters()) / 1e6
model_size_mb = sum(p.element_size() * p.nelement() for p in model.parameters()) / 1024**2

print(f'Params (M): {params:.4f}, Model Size: {model_size_mb:.4f} MB\t')


# FLOPs and GFLOPs Computations
def count_flops(model, input_tensor):
    """
    Counts FLOPs for Conv2d, Linear, and MultiheadAttention layers.
    Multiply-Accumulates: MACs = 2 FLOPs.
    """
    flops = 0
    hooks = []

    def conv_hook(self, input, output):   
        nonlocal flops
        x = input[0]
        batch_size = x.shape[0]
        out_c, out_h, out_w = output.shape[1:]
        in_c = self.in_channels
        k_h, k_w = self.kernel_size
        groups = self.groups

        conv_flops = 2 * batch_size * out_h * out_w * out_c * (in_c // groups * k_h * k_w)
        flops += conv_flops

    def linear_hook(self, input, output):
        nonlocal flops
        x = input[0]
        batch_size = x.shape[0]
        in_features = self.in_features
        out_features = self.out_features

        linear_flops = 2 * batch_size * in_features * out_features
        flops += linear_flops

    def mha_hook(self, input, output):
        """
        input[0]: (B, N, D)
        """
        nonlocal flops
        x = input[0]
        B, N, D = x.shape
        h = self.num_heads
        d = D // h

        # Q, K, V projections: 3 * (2 * B * N * D * D)
        qkv_flops = 3 * (2 * B * N * D * D)

        # Attention: QK^T and Attn*V
        attn_flops = 2 * B * h * N * N * d   # QK^T
        attn_flops += 2 * B * h * N * N * d  # Attn * V

        # Output projection
        out_proj_flops = 2 * B * N * D * D

        flops += qkv_flops + attn_flops + out_proj_flops

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(mha_hook))

    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    return flops


def compute_gflops(model, input_size=(1,3,256,256)):
    flops = []

    def conv_hook(self, input, output):
        batch, Cin, Hin, Win = input[0].shape
        Cout, _, Kh, Kw = self.weight.shape
        Hout, Wout = output.shape[2], output.shape[3]

        if self.groups == Cin:  # depthwise
            ops = 2 * Hout * Wout * Cin * Kh * Kw
        else:
            ops = 2 * Hout * Wout * Cin * Cout * Kh * Kw

        flops.append(ops)

    handles = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))

    dummy = torch.randn(input_size)
    model(dummy)

    for h in handles:
        h.remove()

    gflops = sum(flops)/1e9
    
    return gflops


print(f"================ FLOPs and GFLOPs Computations ========================\t")

input_tensor = torch.randn(1, 3, 256, 256).type(torch.FloatTensor).cuda()


flops = count_flops(model, input_tensor)
gflops = flops / 1e9

print(f"FLOPs: {flops:.4f}, GFLOPs: {gflops:.4f}\t")   


# # Computation of Params (M) and GFLOPs
# with torch.cuda.device(0):
#     macs, params = cps(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)   
#     print('{:<30} {:<8}'.format('Computational Complexity: ', macs))
#     print('{:<30} {:<8}'.format('No. of parameters: ', params)) 
# exit()      


# Datasets and DataLoaders
training_set = CrowdCountingDataset_v3(root_dir=data_path,   
                        split='train_data',
                        train=True,      
                        transform=transforms.Compose([    
                        #transforms.Resize((640, 640)),     # Resize the image of size: 640 x 640
                        #transforms.RandomResizedCrop((224, 224)),    # Random resizing and cropping the image of size: 224 x 224
                        transforms.ToTensor(),             # Convert to [C, H, W] and scales [0, 255] → [0, 1] 
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],      # Normalize image data
                                             std=[0.229, 0.224, 0.225])    
                    ])) 

test_set = CrowdCountingDataset_v3_test(root_dir=data_path,   
                       split='test_data', 
                       train=False,                    
                       transform=transforms.Compose([
                       #transforms.Resize((640, 640)),     # Resize the image of size: 640 x 640
                       transforms.ToTensor(),         # Convert to [C, H, W] and scales [0, 255] → [0, 1]
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],      # Normalize image data       
                                            std=[0.229, 0.224, 0.225])
                  ]))  

# Create DataLoaders
train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
test_loader = DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False)   


# Create a folder to save the checkpoints
checkpoints = '/home/pguha5/Shahbaz/Yogesh_Shahbaz_Crowd_Counting/Journal_2025/\
ShuffMob_CrowdNet/Ablation_Exp10_Backbone_FRM_PFFM_BFLOAT_inference_AVSS_2026/\
Github_PaperID_38_AVSS_2026/weights/\
training01_256x256_val_256x256/saved_model'    

# os.makedirs(checkpoints, exist_ok=True)

checkpoint_path = os.path.join(checkpoints, 'model_checkpoint.pth')   


# Save checkpoint
def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")   

# Resume/Load checkpoint
def load_checkpoint(model, optimizer, scheduler, filename, device):  
    print(f"Loading checkpoint: {filename}")
    
    checkpoint = torch.load(filename, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_metric = checkpoint.get('best_metric', None)
    
    print(f"Resumed from epoch {start_epoch}")
    
    return model, optimizer, scheduler, start_epoch, best_metric


# Define the loss function
#criterion = nn.MSELoss().cuda()
#criterion = nn.MSELoss(size_average=False).cuda()      
#criterion = nn.MSELoss(reduction='sum').cuda()   


#--- Quality metrics: PSNR and SSIM ---
# PSNR
def compute_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr


# SSIM
def ssim(x, y, c1=0.01**2, c2=0.03**2):
    mu_x = x.mean(dim=(-1,-2), keepdim=True)
    mu_y = y.mean(dim=(-1,-2), keepdim=True)

    sigma_x = ((x - mu_x)**2).mean(dim=(-1,-2), keepdim=True)
    sigma_y = ((y - mu_y)**2).mean(dim=(-1,-2), keepdim=True)
    sigma_xy = ((x - mu_x)*(y - mu_y)).mean(dim=(-1,-2), keepdim=True)

    ssim = ((2*mu_x*mu_y + c1)*(2*sigma_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1)*(sigma_x + sigma_y + c2))

    ssim = ssim.mean()  
    return ssim 


def compute_ssim(pred, gt):
    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = pred.mean()
    mu_y = gt.mean()

    sigma_x = pred.var()
    sigma_y = gt.var()
    sigma_xy = ((pred - mu_x)*(gt - mu_y)).mean()

    ssim = ((2*mu_x*mu_y + c1)*(2*sigma_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1)*(sigma_x + sigma_y + c2))

    return ssim     


#--- Inference metrics: Latency and FPS ---
def measure_inference(model, input_tensor, device="cuda", repetitions=10):
    model.eval()
    input_tensor = input_tensor.to(device)

    # Warmup
    for _ in range(repetitions):
        _ = model(input_tensor)  

    torch.cuda.synchronize()
    start_time = time.time()

    _ = model(input_tensor)

    torch.cuda.synchronize()
    end_time = time.time()

    latency = (end_time - start_time) * 1000  # ms
    fps = 1000 / latency

    return latency, fps    


# Composite crowd loss
class CrowdLoss(nn.Module):
    def __init__(self, 
        lambda_mse=1.0,
        lambda_count=0.1,
        lambda_ssim=0.01
        ):
        super(CrowdLoss, self).__init__()
        self.mse = nn.MSELoss()

        self.lambda_mse = lambda_mse
        self.lambda_count = lambda_count
        self.lambda_ssim = lambda_ssim


    def forward(self, est_density_map, gt_density_map):
        """
        est_density_map: (B, 1, H, W) or (B, H, W)
        gt_density_map: (B, 1, H, W) or (B, H, W)
        """
        
        # Ensure both are 4D: (B, 1, H, W)
        if est_density_map.dim() == 3:
            est_density_map = est_density_map.unsqueeze(1)
        if gt_density_map.dim() == 3:
            gt_density_map = gt_density_map.unsqueeze(1)

        # 1) Density map MSE loss
        mse_loss = self.mse(est_density_map, gt_density_map)

        # 2) Count loss (sum over spatial dims)
        est_count = est_density_map.sum(dim=[1,2,3])  # (B,)
        gt_count = gt_density_map.sum(dim=[1,2,3])    # (B,)

        count_loss = torch.abs(est_count - gt_count).mean()

        # 3) SSIM loss
        ssim_loss = 1.0 - ssim(est_density_map, gt_density_map)

        # 4) Total crowd loss
        total_loss = self.lambda_mse * mse_loss + \
            self.lambda_count * count_loss + \
            self.lambda_ssim * ssim_loss

        return total_loss


criterion = CrowdLoss(
    lambda_mse, 
    lambda_count, 
    lambda_ssim
    ).cuda()


# Define the optimizer 
optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
#optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)    
#optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)  

# Optimizer for model2
# optimizer2 = torch.optim.Adam(model.model2.parameters(), lr=initial_lr, weight_decay=weight_decay)


# LR sheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)    
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

# LR sheduler (Warmup + Cosine)
def adjust_lr(optimizer, epoch, epochs, initial_lr):
    warmup_ratio = 0.05      # 5% warmup (use 0.10 for 10%)      
    warmup_epochs = int(epochs * warmup_ratio)

    if epoch < warmup_epochs:
        return initial_lr * (epoch / warmup_epochs)
    else:
        return initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs)/(epochs - warmup_epochs)))   

    lr_scale = float(epoch + 1) / float(warmup_epochs)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr * lr_scale


# Training
def train(model, train_loader, criterion, optimizer, print_freq):    
    model.train()     # Train the model

    running_loss = 0.0
    train_losses = []

    start_time = time.time() 
        
    for i, (img, target) in tqdm(enumerate(train_loader)):  

        img = img.cuda() 
        img = Variable(img)
        #print(img.shape)
            
        target = target.type(torch.FloatTensor).unsqueeze(0)   
        target = target.cuda() 
        target = Variable(target) 
        target = torch.permute(target, (1, 0, 2, 3))  
        #print(target.shape)    
            
        # Forward pass
        outputs = model(img)  
        # outputs = F.interpolate(outputs, size=[target.size(2), target.size(3)], mode="bilinear", align_corners=False)  
        #print(outputs.shape) 

        loss = criterion(outputs, target)

        running_loss += loss.item()
                
        optimizer.zero_grad()   # Reset the gradients
                
        # Backward pass and optimize
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # Gradient clipping
        optimizer.step()      
        
        start_time = time.time() 

        train_loss = running_loss / len(train_loader) 
        train_losses.append(train_loss)
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        if i % print_freq == 0:      
            print(f'Training Loss: {train_loss:.4f} (Epoch Training Loss: {avg_train_loss:.4f})\t')
    
    return avg_train_loss


# Validation
def validate(model, test_loader, criterion):    
    print(f'====================== begin test =====================\t')    
    
    model.eval()      # Evaluate the model
    
    running_loss = 0.0
    val_losses = []
    
    mae = 0
    mse = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for img, target in test_loader:
                
            img = img.cuda()
            img = Variable(img)
            #print(img.shape)
            
            target = target.type(torch.FloatTensor).unsqueeze(0)    
            target = target.cuda()
            #print(target.shape)
            
            # Forward pass    
            outputs = model(img)
            # outputs = F.interpolate(outputs, size=[target.size(2), target.size(3)], mode="bilinear", align_corners=False)
            #print(outputs.shape)
    
            loss = criterion(outputs, target) 
            
            running_loss += loss.item()   
                
            # Calculate the evaluation metrics
            est_count = outputs.detach().cpu().sum().numpy()   # Estimated count
            gt_count = target.detach().cpu().sum().numpy()     # Ground truth (GT) count  
            
            mae += abs(est_count - gt_count)          # Mean absolute error (MAE)
            mse += (est_count - gt_count)**2          # Mean squared error (MSE)
            
        avg_mae = mae / len(test_loader)  
        avg_mse = np.sqrt(mse / len(test_loader))   

        val_loss = running_loss / len(test_loader)  
        val_losses.append(val_loss)
        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f'Val Loss: {val_loss:.4f} (Epoch Val Loss: {avg_val_loss:.4f})\t') 
        print(f'MAE: {avg_mae:.3f}, MSE: {avg_mse:.3f}\t') 
             
    return avg_val_loss, avg_mae, avg_mse 


#---------------------------------------------
# Training loop
#---------------------------------------------
for epoch in tqdm(range(epochs), desc="Training Progress", leave=True):
    print(f"===================== begin training ============================\t")
    
    print('epoch %d, processed %d samples, lr %.10f' % (epoch+1, epoch * len(train_loader.dataset), initial_lr))  

    # ---------- Linear Warmup ----------
    if epoch < warmup_epochs:
        lr_scale = float(epoch + 1) / float(warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = initial_lr * lr_scale
    else:
        scheduler.step()   
    
    # adjust_lr(optimizer, epoch, epochs, initial_lr)
    
    print(f"Epoch: {epoch+1}/{epochs}, LR = {optimizer.param_groups[0]['lr']:.6f}")
    
    train_loss = train(model, train_loader, criterion, optimizer, print_freq)    
    val_loss, val_mae, val_mse = validate(model, test_loader, criterion) 

    # scheduler.step()
    
    # scheduler.step(val_loss)
    # current_lr = scheduler.get_last_lr()[0]  

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\t")   
    

    # ========================= Save the best model ===========================================
    
    # ---------------- Save the best MAE model ----------------  
    if val_mae < best_mae:
        best_mae = val_mae
        best_mse = val_mse                               
            
        best_mae_model = copy.deepcopy(model.state_dict()) 

        best_model_path = os.path.join(checkpoints, 'model_MAE_' + str(epoch+1) + '.pth')
        torch.save(best_mae_model, best_model_path, _use_new_zipfile_serialization=False)

    # best_mae = min(val_mae, best_mae) 
    print(f' * MAE: {val_mae:.3f}, best MAE: {best_mae:.3f}, best MSE: {best_mse:.3f}\t')     

    # save_checkpoint(model, checkpoint_path)  

    
    # ---------------- Save the best validation loss ---------------- 
    if val_loss < best_loss: 
        best_loss = val_loss
        early_stop_counter = 0    
        
        checkpoint_state = {
            'model_state_dict': model.state_dict(),  
        }

        save_checkpoint(checkpoint_state, checkpoint_path)   
    
    else:
        early_stop_counter += 1  
        
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

print(f"Training complete! Best model saved as {checkpoint_path}")   
exit()   

# # Resume if checkpoint exists     
# if os.path.exists(checkpoint_path):
#     model, optimizer, scheduler, start_epoch, best_mae = load_checkpoint(
#         model, optimizer, scheduler, checkpoint_path, device    
#     )   
