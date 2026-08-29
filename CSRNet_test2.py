# Importing the libraries 
import os
import glob
import h5py
from PIL import Image
# from PIL import ImageFilter, ImageDraw
# from PIL import ImageStat
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm  
# # from tqdm.notebook import tqdm      
import shutil                                
import dataset 
import time
from pathlib import Path

import torch 
import torch.nn as nn  
import torch.nn.utils.prune as prune                                    
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable 
# import torch.backends.cudnn as cudnn     
import torchvision
from torchvision import datasets, transforms 
import torchvision.models as models  

# %matplotlib inline   

from model.CSRnet import CoarseNet
from model.CSRnet import CascadedCrowdNet

from data import CrowdCountingDataset2, data_path_gen_SHT
from data import CrowdCountingDataset2, data_path_gen_UCF_CC_50
from data import CrowdCountingDataset2, data_path_gen_UCF_QNRF
 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Device configuration
# cudnn.benchmark = True
device = torch.device('cpu' if 'cpu' else 'cuda')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
print(device) 

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Define the root of dataset                       
root_SHT = './dataset/ShanghaiTech_dataset'     # ShanghaiTech                       
root_UCF_CC_50 = './dataset/UCF_CC_50_dataset'     # UCF_CC_50
root_UCF_QNRF = './dataset/UCF-QNRF_dataset'        # UCF_QNRF

data_path_SHT = data_path_gen_SHT(root_SHT)                    # ShanghaiTech 
data_path_UCF_CC_50 = data_path_gen_UCF_CC_50(root_UCF_CC_50)  # UCF_CC_50
data_path_UCF_QNRF = data_path_gen_UCF_QNRF(root_UCF_QNRF)     # UCF-QNRF

# test_img_paths = data_path_SHT[0][1]      # ShanghaiTech Part_A (SHT_A)
# test_img_paths = data_path_SHT[1][1]      # ShanghaiTech Part_B (SHT_B)
# test_img_paths = data_path_UCF_CC_50      # UCF_CC_50
test_img_paths = data_path_UCF_QNRF[1]    # UCF-QNRF   

#test_img_paths = '/Path/to/ShanghaiTech/dataset/ShanghaiTech_dataset/part_A_final'     

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Model test 
batch_size = 1   

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Define the model
model1 = CoarseNet()  # pretrained model1 (initial density estimator)
#model = model.to(dtype=torch.bfloat16)   # Converting the model to bfloat16 for inference
model1 = model1.cuda()
# model = model.to(device='cpu') 

model2 = nn.ReLU()  # trainable model2 (refiner network)   
model2 = model2.cuda()

model = CascadedCrowdNet(model1, model2)   
model = model.cuda()  

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------  

# Load the checkpoints
#---------------------------------------
# Step 1: Load checkpoint1 → model1
#---------------------------------------
checkpoint1_path = './weights/trained_weights_EfficientNet_B0/\
training01_256x256_val_256x256/saved_model/model_MAE_67.pth'

# checkpoint1_path_MoblieNetV1_0_25 = (
#     "./weights/"
#     " trained_weights_MobileNetV1_0_25/"
#     "training01_256x256_val_256x256/"
#     "saved_model/"
#     "model_MAE_24.pth"  
# )

# checkpoint1_path_MoblieNetV1_0_5 = (
#     "./weights/"
#     " trained_weights_MobileNetV1_0_5/"
#     "training01_256x256_val_256x256/"
#     "saved_model/"
#     "model_MAE_65.pth"  
# )

checkpoint1 = torch.load(checkpoint1_path, map_location=torch.device('cpu'))  # pretrained weights    
# checkpoint1 = torch.load(checkpoint1_path_MoblieNetV1_0_25, map_location=torch.device('cpu'))  # pretrained weights   
# checkpoint1 = torch.load(checkpoint1_path_MoblieNetV1_0_5, map_location=torch.device('cpu'))  # pretrained weights   

#model.model1.load_state_dict(checkpoint['state_dict'])
model.model1.load_state_dict(checkpoint1, strict=True)  

# # If checkpoint contains 'state_dict'
# state_dict1 = checkpoint1.get('state_dict', checkpoint1)   

# # Remove unwanted prefixes if needed (e.g., 'module.')
# new_state_dict1 = {}
# for k, v in state_dict1.items():
#     if k.startswith('model1.'):
#         new_state_dict1[k.replace('model1.', '')] = v
#     else:
#         new_state_dict1[k] = v

# # Load only into model1
# model.model1.load_state_dict(new_state_dict1, strict=False)   

# for p in model.model1.parameters(): 
#     p.requires_grad = False  # freeze model1    

model.eval()     

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Data Augmentation
transform = transforms.Compose([
                     #transforms.Resize((224, 224)),     # Resize the image of size: 224 x 224
                     transforms.ToTensor(),         # Convert the image to a tensor  
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],      # Normalize image data       
                                          std=[0.229, 0.224, 0.225])    
])  
                                                             

# Plotting the density map: Create a function for plotting the density map                                      
def plot_density_map(original_image, gt_density_map, est_density_map, gt_count, est_count): 
    # Convert tensors to numpy arrays
    # Ensuring original image is a PyTorch tensor                                              
    if not isinstance(original_image, torch.Tensor):                                                  
        original_image = torch.tensor(original_image)   
  
    original_image = original_image.squeeze().detach().cpu().numpy()
    
    #======================================================================================
    
    # Ensuring gt_density_map is a PyTorch tensor                                              
    if not isinstance(gt_density_map, torch.Tensor):                                                  
        gt_density_map = torch.tensor(gt_density_map)                            
    
    gt_density_map = gt_density_map.squeeze().detach().cpu().numpy()  
    
    #======================================================================================
    
    # Ensuring est_density_map is a PyTorch tensor
    if not isinstance(est_density_map, torch.Tensor):
        est_density_map = torch.tensor(est_density_map)
    
    est_density_map = est_density_map.squeeze().detach().cpu().numpy() 
    
       
    # Create the subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))      
    
    #========================================================================================  
  
    # Plot the original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image') 
    # axes[0].axis('off')  # Turn off axis lines, ticks, and labels for original image subplot
    
    #========================================================================================
   
    # Plot the ground truth (GT) density map
    axes[1].imshow(gt_density_map, cmap='jet')
    axes[1].set_title('Ground Truth (GT) Density Map')
    # axes[1].axis('off')  # Turn off axis lines, ticks, and labels for GT density map subplot  
    
    #========================================================================================

    # Plot the estimated density map
    axes[2].imshow(est_density_map, cmap='jet')
    axes[2].set_title('Estimated Density Map')
    # axes[2].axis('off')  # Turn off axis lines, ticks, and labels for estimated density map subplot

    # Print the GT count and estimated count on GT density and estimated density maps
    gt_text = f"GT Count: {gt_count:.4f}"
    est_text = f"Estimated Count: {est_count:.4f}"    
    
    #========================================================================================
    
    font_size = 10
    
    #========================================================================================

    # Print the GT count on the right upper corner of GT density map
    # Get the dimensions of the GT density map
    gt_height, gt_width = gt_density_map.shape

    # Set the position of the GT count (right upper corner)
    gt_count_x_pos = gt_width - 350  # Adjust the value as needed to position the GT count horizontally
    gt_count_y_pos = 50  # Adjust the value as needed to position the GT count vertically

    # Add the GT count to the GT density map plot
    # axes[1].text(gt_count_x_pos, gt_count_y_pos, gt_text, color='white', fontsize=font_size, alpha=0.98)
    # axes[1].text(gt_count_x_pos, gt_count_y_pos, gt_text, color='white', fontsize=font_size, bbox=dict(facecolor='black', alpha=0.5))
    
    #==========================================================================================
    
    # Print the estimated count on the right upper corner of estimated density map
    # Get the dimensions of the estimated density map
    est_height, est_width = est_density_map.shape

    # Set the position of the estimated count (right upper corner) 
    est_count_x_pos = est_width - 350  # Adjust the value as needed to position the estimated count horizontally
    est_count_y_pos = 50  # Adjust the value as needed to position the estimated count vertically     

    # Add the estimated count to the estimated density map plot
    # axes[2].text(est_count_x_pos, est_count_y_pos, est_text, color='white', fontsize=font_size, alpha=0.98)     
    # axes[2].text(est_count_x_pos, est_count_y_pos, est_text, color='white', fontsize=font_size, bbox=dict(facecolor='black', alpha=0.5))       
    
    #=========================================================================================

    # Hide axis ticks and labels for all the subplots    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])   

    # Adjust spacing between the subplots       
    plt.tight_layout()      
    
    # Show the plot     
    plt.show()   

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    
# Test   
def test(model, batch_size, test_img_paths):
    print('begin test')    
                                           
    model.eval()      # Evaluate the model
    mae = 0
    mse = 0 
    
    test_set = CrowdCountingDataset2(img_paths=test_img_paths,                
                       transform=transforms.Compose([
                       transforms.ToTensor(),         # Convert the image to a tensor  
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],      # Normalize image data       
                                            std=[0.229, 0.224, 0.225])
                  ]))
         
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)        
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, (image, gt_density_map) in enumerate(test_loader):
                
            image = image.cuda()
            image = Variable(image)
            
            gt_density_map = gt_density_map.type(torch.FloatTensor) 
            gt_density_map = gt_density_map.cuda()
            
            # Forward pass    
            est_density_map = model(image) 
        
            # Calculate the evaluation metrics 
            est_count = est_density_map.detach().cpu().sum().numpy()   # Estimated count
            gt_count = gt_density_map.detach().cpu().sum().numpy()     # Ground truth (GT) count             
        
            mae += abs(est_count - gt_count)          # Mean absolute error (MAE)
            mse += (est_count - gt_count)**2          # Mean squared error (MSE)
            
        avg_mae = mae / len(test_loader)  
        avg_mse = np.sqrt(mse / len(test_loader))     

        print(f'MAE: {avg_mae:.3f}, MSE: {avg_mse:.3f}\t')      
        
        #=======================================================================================
        
        # Measure the inference time on GPU    
        inference_time_gpu = (time.time() - start_time) * 1000  # Convert to milliseconds (ms)
    
        # Calculate the inference speed in frames per second (fps)
        inference_speed_gpu = 1000.0 / inference_time_gpu
        
        print(f'Inference time on GPU: {inference_time_gpu:.2f} ms, Inference speed on GPU: {inference_speed_gpu:.2f} fps\t')  
        
        #========================================================================================
        
        # # Plotting the density map
        # plot_density_map(image, gt_density_map, est_density_map, gt_count, est_count)
        
        # image = image.squeeze()
        # gt_density_map = gt_density_map.squeeze()
        # est_density_map = est_density_map.squeeze()
        
        # print('Image size: ', image.shape)
        # print('Ground truth (GT) density map size: ', gt_density_map.shape)
        # print('Estimated density map size: ', est_density_map.shape)
        
        # print("Ground Truth (GT) Count: ", gt_count) 
        # print("Estimated Count: ", est_count) 
        
        return avg_mae, avg_mse    

#============================================================================================================================================================================

test(model, batch_size, test_img_paths)   
# exit()   
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create the functions for resizing the image and density map
def resize_image(image_path, new_size):
    image = transform(Image.open(image_path).convert('RGB')).cuda()
    resized_image = image.resize(new_size, Image.BICUBIC) 
    return resized_image
    
def resize_density_map(density_map_path, new_size):
    gt_file = h5py.File(density_map_path, 'r')
    gt_density_map = np.asarray(gt_file['density'])
    resized_density_map = gt_density_map.resize(new_size, Image.BICUBIC)       
    return resized_density_map
    
# Plotting the image and density maps
# Specify the size for the subplot images
# subplot_size = (512, 1024)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot the image
image_path = './dataset/UCF-QNRF_dataset/test/images/img_0295.jpg'  

# Load the image
# image = plt.imread(image_path)
image = np.asarray(Image.open(image_path).convert('RGB')) 
image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC) 
print("Original Image")    

# Display the image without axes
plt.imshow(image_resized)
plt.title("Original Image") 
plt.axis('off')  # Turn off the axis
# plt.legend()
plt.show()   

#=====================================================================================================

image_path = './dataset/UCF-QNRF_dataset/test/images/img_0295.jpg'             
# image = transform(Image.open(image_path).convert('RGB')).cuda() 
image = np.asarray(Image.open(image_path).convert('RGB'))
image_resized_np = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC) 
print('Image size: ', image_resized_np.shape)   

# image = resize_image(image_path, subplot_size)         

#=====================================================================================================

gt_density_map_path = './dataset/UCF-QNRF_dataset/test/ground_truth/GT_h5_FGK_sigma_4/img_0295.h5'    
gt_file = h5py.File(gt_density_map_path, 'r')                       
gt_density_map = np.asarray(gt_file['density']) 

# gt_density_map_resized = cv2.resize(gt_density_map, (image_resized.shape[0] // 8, image_resized.shape[1] // 8), interpolation=cv2.INTER_CUBIC) * 64 
gt_resize = (image_resized_np.shape[1] // 8, image_resized_np.shape[0] // 8) 
gt_density_map_resized = cv2.resize(gt_density_map, (gt_resize[0], gt_resize[1]), interpolation=cv2.INTER_CUBIC) 

# Scale the density map to preserve the count
scale = (gt_density_map.shape[0] * gt_density_map.shape[1]) / (gt_density_map_resized.shape[0] * gt_density_map_resized.shape[1]) 
gt_density_map_resized *= scale  

print('Ground truth (GT) density map size: ', gt_density_map_resized.shape)                
# gt_density_map = resize_density_map(gt_density_map_path, subplot_size)   
                                                                                        
gt_count = np.sum(gt_density_map_resized)
print("Ground truth (GT) count: ", gt_count)   

#=====================================================================================================

image_resized = Image.fromarray(image_resized_np)
image_resized = transform(image_resized).cuda() 
est_density_map = model(image_resized.unsqueeze(0))

est_density_map = est_density_map.squeeze().detach().cpu().numpy()
print('Estimated density map size: ', est_density_map.shape) 
est_count = np.sum(est_density_map)
print("Estimated count: ", est_count)  

count_error = np.abs(est_count - gt_count) 
print("Count error: ", count_error)  

#=============================================================================================================================================================================

# Plotting the density map
plot_density_map(image_resized_np, gt_density_map_resized, est_density_map, gt_count, est_count) 
exit()  

