# Importing the libraries
import os
import glob
import h5py 
import scipy 
import scipy.io as io 
import scipy.spatial as spatial                    
from scipy.ndimage import gaussian_filter      
from PIL import Image
import csv
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from image import *

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
# Density map generation: Create a function to generate the density maps using a geometry-adaptive kernel (GAK) for the images
#-------------------------------------------------------------------------------------------------------------------------------------------
def gaussian_filter_density(gt):                           
    print(gt.shape)                           
    density = np.zeros(gt.shape, dtype=np.float32)          
    gt_count = np.count_nonzero(gt)                           
    
    if gt_count == 0:                               
        return density  

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))                           
    leaf_size = 2048

    # Build kd tree                           
    tree = spatial.KDTree(pts.copy(), leafsize=leaf_size)                           
    
    # Query kd tree                           
    distances, locations = tree.query(pts, k=4)  
    print('generate density...') 

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32) 
        pt2d[pt[1], pt[0]] = 1. 

        if gt_count > 1:                                   
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1                               
        else:                                   
            sigma = np.average(np.array(gt.shape))/2./2. # case: 1 point   

        density += gaussian_filter(pt2d, sigma, mode='constant') 

    print('done.')                           
    return density 

#===========================================================================================================================================================

#--------------------------------------------------------------------------------------------------------------------------------------------
# Density map generation: Create a function to generate the density maps using a fixed Gaussian kernel (FGK) for the images
#--------------------------------------------------------------------------------------------------------------------------------------------
def generate_density_map_fixed_gaussian_kernel(img, points, kernel_size=15, sigma=10.0):
    '''
    Generate ground truth (GT) density map for crowd counting.
    
    Parameters:
    img: input image.
    points: annotated pedestrian's/head's position like [row, col].
    kernel_size: the fixed size of gaussian kernel, must be an odd number.
    sigma: the sigma of gaussian kernel.

    Return:
    density_map: density map. 
    
    '''
    
    # Create a function for defining the Gaussian kernel
    def guassian_kernel(size, sigma):
        rows = size[0]                    # Mind that the size must be an odd number.
        cols = size[1]
        mean_x = int((rows-1)/2)
        mean_y = int((cols-1)/2)
        
        # Create an empty gaussian kernel function with the same shape as the input image
        f = np.zeros(size)
        
        for x in range(0, rows):
            for y in range(0, cols):
                mean_x2 = (x-mean_x)**2
                mean_y2 = (y-mean_y)**2
                
                f[x, y] = (1.0/(2.0 * np.pi * sigma**2)) * np.exp((mean_x2 + mean_y2)/(-2.0 * sigma**2))  
        
        return f

    
    [rows, cols] = [img.shape[0], img.shape[1]]
    
    # Create an empty density map with the same shape as the input image
    density_map = np.zeros([rows, cols])
    f = guassian_kernel([kernel_size, kernel_size], sigma)       # Generate a gaussian kernel with the fixed size.
    normed_f = (1.0/f.sum()) * f                                 # Normalization for each head.
    
    print('generate density map...') 
    
    if len(points) == 0:
        return density_map
    else:
        for pt in points:
            r, c = int(pt[0]), int(pt[1])
            
            if r >= rows or c >= cols:
                continue
            
            for x in range(0, f.shape[0]):
                for y in range(0, f.shape[1]):
                    if x + ((r+1) - int((f.shape[0]-1)/2)) < 0 or x + ((r+1) - int((f.shape[0]-1)/2)) > rows-1 \
                    or y + ((c+1) - int((f.shape[1]-1)/2)) < 0 or y + ((c+1) - int((f.shape[1]-1)/2)) > cols-1:
                        continue
                    else:
                        density_map[x + ((r+1) - int((f.shape[0]-1)/2)), y + ((c+1) - int((f.shape[1]-1)/2))] += normed_f[x,y]
    
    print('done.') 
    return density_map

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------
# Generate the ground truth (GT) values for images in ShanghaiTech/UCF_CC_50/UCF-QNRF/WorldExpo'10/NWPU-Crowd/JHU-CROWD++ dataset
#------------------------------------------------------------------------------------------------------------------------------------------------
# Root of the ShanghaiTech/UCF_CC_50/UCF-QNRF/WorldExpo'10/NWPU-Crowd/JHU-CROWD++ dataset
root = './dataset/ShanghaiTech_dataset' 

#------------------------------------------------------------------------------------------
# ShanghaiTech dataset:
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')                       
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')                       
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')                       
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images') 
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
# # UCF_CC_50 dataset:
# image_path = os.path.join(root, 'UCF_CC_50', 'images') 
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
# # WorldExpo'10 dataset:
# train_path = os.path.join(root, 'train', 'img')                       
# test_path = os.path.join(root, 'test/test_New/S1', 'img')   
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
# # UCF-QNRF/NWPU-Crowd/JHU-CROWD++ dataset:
# train_path = os.path.join(root, 'train', 'images')  
# val_path = os.path.join(root, 'val', 'images')                       
# test_path = os.path.join(root, 'test', 'images')                       
#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
# # ShanghaiTech dataset:
# Setting the image locations 
train_path_sets_A = [part_A_train]  
test_path_sets_A = [part_A_test]  

train_path_sets_B = [part_B_train]                         
test_path_sets_B = [part_B_test]

# Define the train image path for ShanghaiTech Part_A
train_img_paths_A = []
for path in train_path_sets_A: 
    for img_path in glob.glob(os.path.join(path, '*.jpg')): 
        train_img_paths_A.append(img_path) 

# Define the test image path for ShanghaiTech Part_A
test_img_paths_A = []
for path in test_path_sets_A: 
    for img_path in glob.glob(os.path.join(path, '*.jpg')): 
        test_img_paths_A.append(img_path) 

# Define the train image path for ShanghaiTech Part_B
train_img_paths_B = []
for path in train_path_sets_B: 
    for img_path in glob.glob(os.path.join(path, '*.jpg')): 
        train_img_paths_B.append(img_path) 

# Define the test image path for ShanghaiTech Part_B
test_img_paths_B = []
for path in test_path_sets_B: 
    for img_path in glob.glob(os.path.join(path, '*.jpg')): 
        test_img_paths_B.append(img_path) 
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# # UCF_CC_50 dataset:
# # Setting the image locations                      
# path_sets = [image_path] 

# # Define the image path
# img_paths = [] 
# for path in path_sets:                           
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):             
#         img_paths.append(img_path)
#-----------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------
# # UCF-QNRF/WorldExpo'10/NWPU-Crowd/JHU-CROWD++ dataset: 
# # Setting the image locations                      
# train_path_sets = [train_path] 
# val_path_sets = [val_path] 
# test_path_sets = [test_path]

# # Define the train image path
# train_img_paths = [] 
# for path in train_path_sets:                           
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):             
#         train_img_paths.append(img_path)

# # Define the val image path
# val_img_paths = [] 
# for path in val_path_sets:                           
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):             
#         val_img_paths.append(img_path)                               

# # Define the test image path      
# test_img_paths = [] 
# for path in test_path_sets:                           
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):             
#         test_img_paths.append(img_path)
#-------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------
# Creating a density map for the ShanghaiTech/UCF_CC_50/UCF-QNRF/WorldExpo'10/NWPU-Crowd/JHU-CROWD++ dataset 
#---------------------------------------------------------------------------------------------------------------------------------------------------
for img_path in train_img_paths:                           
    print(img_path) 

    #---------------------------------------------------------------------------------------------------------------------
    # ShanghaiTech dataset:
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'))                           
    img = plt.imread(img_path)
    #---------------------------------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------------------------------------------------------
    # # UCF_CC_50 dataset:
    # mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('img_', 'ann_'))                           
    # img = plt.imread(img_path)
    #----------------------------------------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------------------------------------
    # # UCF-QNRF dataset:
    # mat = io.loadmat(img_path.replace('.jpg', '_ann.mat').replace('images', 'ground_truth'))                           
    # img = plt.imread(img_path)  
    #-----------------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------------------
    # # WorldExpo'10 dataset:
    # # Construct the output path for the CSV file
    # csv_path = img_path.replace('.jpg', '.csv').replace('img', 'den')   

    # # Read a csv file into DataFrame
    # df_csv = pd.read_csv(csv_path, header=None) 
    
    # # Convert the DataFrame to a NumPy array
    # annotations = df_csv.values.astype(dtype=np.float32) 
    # #annotations = df_csv.to_numpy()
    
    # img = plt.imread(img_path) 
    
    # h, w = img.shape[:2]  
    #-----------------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------------------
    # # NWPU-Crowd dataset:
    # mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth'))                          
    # img = plt.imread(img_path)  

    # h, w = img.shape[:2]  
    #-------------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------------
    # # JHU-CROWD++ dataset:
    # # Construct the output path for the TXT file
    # txt_path = img_path.replace('.jpg', '.txt').replace('images', 'gt')
    
    # # Read a TXT file into numpy array
    # txt = np.loadtxt(txt_path)  
    
    # img = plt.imread(img_path) 

    # h, w = img.shape[:2]  
    #--------------------------------------------------------------------------------------------------------------------------
    
#===========================================================================================================================================================   

    #----------------------------------------------------------------------------------------------------------------------------------------------
    # Generating density map using geometry-adaptive kernel (GAK):
    #----------------------------------------------------------------------------------------------------------------------------------------------
    k = np.zeros((img.shape[0], img.shape[1]))  
    
    # Extracting the head annotations 
    #------------------------------------------------------------------------------------
    # ShanghaiTech:                      
    gt = mat["image_info"][0,0][0,0][0]       # 1546 person * 2 (col, row) 
    #------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------
    # # UCF_CC_50/UCF-QNRF:
    # gt = mat["annPoints"]        # 1546 person * 2 (col, row) 
    #------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------
    # # NWPU-Crowd:
    # gt = mat["annPoints"]        # 1546 person * 2 (col, row) 
    #------------------------------------------------------------------------------------
   
    for i in range(0, len(gt)):                               
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:                                   
            k[int(gt[i][1]), int(gt[i][0])] = 1  

    k = gaussian_filter_density(k) 

    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground-truth/GT_h5_GAK'), 'w') as hf:                                            
        hf['density'] = k  
     
#=============================================================================================================================================================    

    #-----------------------------------------------------------------------------------------------------------------------------------------------
    # Generating density map using fixed Gaussian kernel (FGK):
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    
    # Extracting the head annotations
    #--------------------------------------------------------------------------------------
    # ShanghaiTech:
    pts = mat["image_info"][0,0][0,0][0]        # 1546 person * 2 (col, row)  
    #--------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------
    # # UCF_CC_50/UCF-QNRF:
    # pts = mat["annPoints"]        # 1546 person * 2 (col, row) 
    #--------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------
    # # NWPU-Crowd:
    # pts = mat["annPoints"]        # 1546 person * 2 (col, row) 
    #--------------------------------------------------------------------------------------
    
    points = []

    #-------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------ 
    # ShanghaiTech/UCF_CC_50/UCF-QNRF dataset:
    #------------------------------------------------
    for pt in pts:
        x, y = pt[0], pt[1]
        points.append([y, x])             # Convert (col, row) to (row, col) 

        density_map = generate_density_map_fixed_gaussian_kernel(img, points)    
    
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground-truth/GT_h5_FGK_sigma_10'), 'w') as hf:                                                
            hf['density'] = density_map   
    #-------------------------------------------------------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------
    # WorldExpo'10 dataset:
    #----------------------------------------------
    # height, width = annotations.shape
    # for y in range(height):
    #     for x in range(width):
    #         if annotations[y, x] > 0:  # Annotation value > 0 for the head annotation and annotation value = 0 for non-head annotation
    #             points.append([x, y])  # Store as [x, y]
    
    #             density_map = generate_density_map_fixed_gaussian_kernel(img, points)    
    
    #             with h5py.File(img_path.replace('.jpg', '.h5').replace('img', 'den/GT_h5_FGK_sigma_4'), 'w') as hf:                                      
    #                 hf['density'] = density_map 
    
    #         else: 
    #             # Check the empty annotations
    #             # Generate a zero density map
    #             print('Generate a zero density map')
    #             density_map = np.zeros((h, w), dtype=np.float32) 
    #             print('Density map done.')    

    #             with h5py.File(img_path.replace('.jpg', '.h5').replace('img', 'den/GT_h5_FGK_sigma_4'), 'w') as hf:                                      
    #                 hf['density'] = density_map  
    #----------------------------------------------------------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------
    # NWPU-Crowd dataset:
    #--------------------------------------------------
    # # Check for the empty annotations
    # # if 'annPoints' in mat and len(mat['annPoints']) == 0:
    # if len(pts) == 0:
    #     # Generate a zero density map
    #     print('Generate a zero density map')
    #     density_map = np.zeros((h, w), dtype=np.float32) 
    #     print('Density map done.')

    #     with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth/GT_h5_FGK_sigma_10'), 'w') as hf:                                         
    #         hf['density'] = density_map  
    # else:
    #     for pt in pts:
    #         x, y = pt[0], pt[1]   
    #         points.append([y, x])             # Convert (col, row) to (row, col)   
        
    #         density_map = generate_density_map_fixed_gaussian_kernel(img, points)    
    
    #         with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth/GT_h5_FGK_sigma_10'), 'w') as hf:                                         
    #             hf['density'] = density_map  
    #----------------------------------------------------------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------
    # JHU-CROWD++ dataset:
    #----------------------------------------------------
    # # Accessing the data
    # # Check for the empty annotations
    # if txt.ndim >= 2:
    #     # If txt is >= 1D
    #     x_coords = txt[:, 0]          # Assuming the first column is x coordinates
    #     y_coords = txt[:, 1]          # Assuming the second column is y coordinates
    #     # You can continue this pattern for more dimensions if needed
         
    #     pts = [x_coords, y_coords]  
   
    #     # Convert tuple elements to strings and join them into a single string
    #     pt_x = ''.join(str(item) for item in pts[0].shape)
    #     pt_y = ''.join(str(item) for item in pts[1].shape)    

    #     ann_points = []

    #     for i in range(int(pt_x)):                                       
    #         x, y = pts[0][i], pts[1][i]    
    #         ann_points.append([y, x])             # Convert (col, row) to (row, col) 
    
    #         density_map = generate_density_map_fixed_gaussian_kernel(img, ann_points)    
    
    #         with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'gt/GT_h5_FGK_sigma_10'), 'w') as hf:                                          
    #             hf['density'] = density_map  

    # else: 
    #     # Check for the empty annotations
    #     # If txt is <= 1D
    #     # Generate a zero density map
    #     print('Generate a zero density map')
    #     density_map = np.zeros((h, w), dtype=np.float32)
    #     print('Density map done.')

    #     with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'gt/GT_h5_FGK_sigma_10'), 'w') as hf:                                          
    #         hf['density'] = density_map 
    #----------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Plot a sample image and its ground truth density map
#-----------------------------------------------------------------------------
# plt.imshow(Image.open(test_img_paths[0])) 
# plt.show() 

# # Show the density map corresponding to the image
# gt_file = h5py.File(test_img_paths[0].replace('.jpg', '.h5').replace('images', 'ground-truth/GT_h5_FGK_sigma_15'), 'r') 
# ground_truth = np.asarray(gt_file['density']) 
# plt.imshow(ground_truth, cmap=CM.jet)
# plt.title("Ground Truth (GT) Density Map")    
# plt.show()   

# # Count the no. of people present in this image
# gt_count = np.sum(ground_truth)
# print("GT count: ", gt_count)
# exit()         

# # Show the density map corresponding to the image
# # Density map
# file_path = './dataset/ShanghaiTech_dataset/part_A_final/train_data/ground-truth/GT_h5_FGK_sigma_15/IMG_100.h5'
# gt_file = h5py.File(file_path, 'r')
# ground_truth = np.asarray(gt_file['density'])
# plt.imshow(ground_truth, cmap=CM.CMRmap)
# plt.title("Ground Truth (GT) Density Map")
# plt.show()   

# gt_count = np.sum(ground_truth)  
# print("GT count: ", gt_count)   

# # Open the image corresponding to the density map
# file = './dataset/ShanghaiTech_dataset/part_A_final/train_data/ground-truth/GT_h5_FGK_sigma_15/IMG_100.h5'  
# image = Image.open(file.replace('.h5', '.jpg').replace('ground-truth/GT_h5_FGK_sigma_15', 'images'))
# plt.imshow(image)
# plt.title("Crowd Image")   
# plt.show()   
