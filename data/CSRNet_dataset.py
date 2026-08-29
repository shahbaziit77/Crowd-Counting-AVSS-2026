# Importing the libraries  
import os
import glob
import h5py 
import scipy.io as io
import random
from PIL import Image
import numpy as np
#import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.cm as CM       
import cv2
from image import *     

import torch  
#import torch.nn as nn  
from torch.utils.data import Dataset
#from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create a custom crowd counting dataset class    
class CrowdCountingDataset2(Dataset):
    def __init__(self, img_paths, transform=None):
        self.nSamples = len(img_paths)
        self.img_paths = img_paths
        self.transform = transform
    
    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]  
        img, target = load_data2(img_path)   
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target     
  
#====================================================================================================================================================

    
# Data loading: Create a function for loading the data
def load_data2(img_path):   
    # gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth/GT_h5_GAK')            # ShanghaiTech Part_A (SHT_A)
    # gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth/GT_h5_FGK_sigma_15')   # ShanghaiTech Part_B (SHT_B)  
    # gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth/GT_h5_GAK')            # UCF_CC_50
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth/GT_h5_FGK_sigma_4')    # UCF-QNRF   

    
    # Load the image
    #img = Image.open(img_path).convert('RGB')
    img = np.array(Image.open(img_path).convert('RGB')) 
    
    # Load an RGB image and convert to numpy array
    #img = np.asarray(Image.open(img_path).convert('RGB')) 
    
    #img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load the density map
    #with h5py.File(gt_path, 'r') as f:
        #target = np.asarray(f['density'])
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])      
    original_count = np.sum(target)
    #print(original_count)  
    
    #======================================================================================================
    
    # Resize the RGB image
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)      
    
    #=======================================================================================================
    
    # Resize the density map
    target_resized = cv2.resize(target, (img_resized.shape[1] // 8, img_resized.shape[0] // 8), interpolation=cv2.INTER_CUBIC) 
    
    # Scale the density map to preserve the count
    scale = (target.shape[0] * target.shape[1]) / (target_resized.shape[0] * target_resized.shape[1]) 
    target_resized *= scale  
    #print(np.sum(target_resized))  
    
    target = torch.from_numpy(target_resized).squeeze(0)    # [1, H, W]  
    img = Image.fromarray(img_resized)  
    
    return img, target   
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------

# Create a custom crowd counting dataset class 
class CrowdCountingDataset_v3(Dataset):
    def __init__(self, root_dir, split='train', train=True, transform=None):
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.target_dir = os.path.join(root_dir, split, 'ground-truth/GT_h5_GAK')

        self.image_paths = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.target_paths = sorted([f for f in os.listdir(self.target_dir) if f.endswith('.h5')])

        assert len(self.image_paths) == len(self.target_paths), "Mismatch in dataset size"
        
        self.train = train
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess the image
        img_path = os.path.join(self.image_dir, self.image_paths[idx])    
        img = np.array(Image.open(img_path).convert('RGB')) 
        
        # Convert to grayscale
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize the image
        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)      
        
        # img = img.astype(np.float32)/255.0
        
        # Load the density map
        base_name = os.path.splitext(self.image_paths[idx])[0]
        gt_path = os.path.join(self.target_dir, base_name + '.h5')
        #with h5py.File(gt_path, 'r') as f:
            #target = np.asarray(f['density'])
        gt_file = h5py.File(gt_path, 'r')
        target = np.asarray(gt_file['density'])    

        # Resize the density map
        target_resized = cv2.resize(target, (img_resized.shape[1] // 8, img_resized.shape[0] // 8), interpolation=cv2.INTER_CUBIC)
        
        # Scale the density map to preserve the count
        scale = (target.shape[0] * target.shape[1]) / (target_resized.shape[0] * target_resized.shape[1]) 
        target_resized *= scale 
        
        #==========================================================================
        
        # Data Augmentation:
        if self.train:
            # Random horizontal flip
            if np.random.rand() > 0.5:
            #if random.random() > 0.5:
                img_resized = np.fliplr(img_resized).copy()
                target_resized = np.fliplr(target_resized).copy()  
        
        #===========================================================================
        
        target = torch.from_numpy(target_resized).squeeze(0)    # [1, H, W]  
        img = Image.fromarray(img_resized)  
    
        if self.transform is not None:
            img = self.transform(img)

        return img, target  

#==============================================================================================================================================

# Create a custom crowd counting dataset class 
class CrowdCountingDataset_v3_test(Dataset):
    def __init__(self, root_dir, split='train', train=True, transform=None):
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.target_dir = os.path.join(root_dir, split, 'ground-truth/GT_h5_GAK')

        self.image_paths = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.target_paths = sorted([f for f in os.listdir(self.target_dir) if f.endswith('.h5')])

        assert len(self.image_paths) == len(self.target_paths), "Mismatch in dataset size"
        
        self.train = train
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess the image
        img_path = os.path.join(self.image_dir, self.image_paths[idx])    
        img = np.array(Image.open(img_path).convert('RGB')) 
        
        # # Convert to grayscale
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize the image
        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)      
        
        # img = img.astype(np.float32)/255.0
        
        # Load the density map
        base_name = os.path.splitext(self.image_paths[idx])[0]
        gt_path = os.path.join(self.target_dir, base_name + '.h5')
        #with h5py.File(gt_path, 'r') as f:
            #target = np.asarray(f['density'])
        gt_file = h5py.File(gt_path, 'r')
        target = np.asarray(gt_file['density'])

        # Resize the density map
        target_resized = cv2.resize(target, (img_resized.shape[1] // 8, img_resized.shape[0] // 8), interpolation=cv2.INTER_CUBIC)   
        
        # Scale the density map to preserve the count
        scale = (target.shape[0] * target.shape[1]) / (target_resized.shape[0] * target_resized.shape[1]) 
        target_resized *= scale 
        
        #==========================================================================
        
        # Data Augmentation:
        if self.train:
            # Random horizontal flip
            if np.random.rand() > 0.5:
            #if random.random() > 0.5:
                img_resized = np.fliplr(img_resized).copy()
                target_resized = np.fliplr(target_resized).copy()   

        #===========================================================================
        
        target = torch.from_numpy(target_resized).squeeze(0)    # [1, H, W]  
        img = Image.fromarray(img_resized)  
    
        if self.transform is not None:
            img = self.transform(img)

        return img, target  

#-----------------------------------------------------------------------------------------------------------------------------------------------

# Create a function for the data path generation of ShanghaiTech dataset
def data_path_gen_SHT(paths):

    part_A_train = os.path.join(paths, 'part_A_final/train_data', 'images')   
    part_A_test = os.path.join(paths, 'part_A_final/test_data', 'images')
    part_B_train = os.path.join(paths, 'part_B_final/train_data', 'images')           
    part_B_test = os.path.join(paths, 'part_B_final/test_data', 'images')                             
    
    # Generate the ground truth (GT) values for images in ShanghaiTech Part_A 
    path_sets = [part_A_train, part_A_test]
    train_path_sets_A = [part_A_train]
    test_path_sets_A = [part_A_test]
                                       
    # Setting the image locations   
    img_paths = [] 
    for path in path_sets:                           
        for img_path in glob.glob(os.path.join(path, '*.jpg')):             
            img_paths.append(img_path)

    # Define the train image path
    train_img_paths_A = []
    for path in train_path_sets_A: 
        for img_path in glob.glob(os.path.join(path, '*.jpg')): 
            train_img_paths_A.append(img_path)                           

    # Define the test image path 
    test_img_paths_A = []
    for path in test_path_sets_A: 
        for img_path in glob.glob(os.path.join(path, '*.jpg')): 
            test_img_paths_A.append(img_path)                           

    # Similarly, generate the ground truth (GT) values for ShanghaiTech Part_B   
    path_sets = [part_B_train, part_B_test] 
    train_path_sets_B = [part_B_train]                        
    test_path_sets_B = [part_B_test]  

    # Setting the image locations    
    img_paths = []
    for path in path_sets:                           
        for img_path in glob.glob(os.path.join(path, '*.jpg')):                               
            img_paths.append(img_path)  

    # Define the train image path
    train_img_paths_B = []
    for path in train_path_sets_B: 
        for img_path in glob.glob(os.path.join(path, '*.jpg')): 
            train_img_paths_B.append(img_path)
        
    # Define the test image path 
    test_img_paths_B = []
    for path in test_path_sets_B: 
        for img_path in glob.glob(os.path.join(path, '*.jpg')): 
            test_img_paths_B.append(img_path)

    return ([train_img_paths_A, test_img_paths_A], [train_img_paths_B, test_img_paths_B])   
    
#==================================================================================================================================================================================    
    
# Create a function for the data path generation of UCF_CC_50 dataset
def data_path_gen_UCF_CC_50(paths):

    image_path = os.path.join(paths, 'UCF_CC_50', 'images')     
    
    # Generate the ground truth (GT) values for images in UCF_CC_50                      
    path_sets = [image_path] 
    
    # Setting the image locations  
    # Define the image path
    img_paths = [] 
    for path in path_sets:                           
        for img_path in glob.glob(os.path.join(path, '*.jpg')):             
            img_paths.append(img_path)
  
    return img_paths

#==================================================================================================================================================================================    
    
# Create a fuction for the data path generation of UCF-QNRF dataset    
def data_path_gen_UCF_QNRF(paths):   
    
    train_path = os.path.join(paths, 'train', 'images')  
    test_path = os.path.join(paths, 'test', 'images')   
    
    # Generate the ground truth (GT) values for images in UCF-QNRF                     
    train_path_sets = [train_path] 
    test_path_sets = [test_path]
    
    # Setting the image locations 
    # Define the train image path
    train_img_paths = [] 
    for path in train_path_sets:                           
        for img_path in glob.glob(os.path.join(path, '*.jpg')):             
            train_img_paths.append(img_path)

    # Define the test image path      
    test_img_paths = [] 
    for path in test_path_sets:                           
        for img_path in glob.glob(os.path.join(path, '*.jpg')):             
            test_img_paths.append(img_path) 
            
    return ([train_img_paths, test_img_paths])      
 
                       
