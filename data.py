import os
import numpy as np
import cv2
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import re


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split('_mask', s)]


def load_data(path_image, path_mask):
  
    image_names = sorted(glob(path_image + '*.jpeg') + glob(path_image + '*.jpg') + glob(path_image + '*.png'), key=natural_sort_key)
   
    mask_names = sorted(glob(path_mask + '*.jpeg') + glob(path_mask +'*.jpg'), key=natural_sort_key) + sorted(glob(path_mask +'*.png'), key=natural_sort_key)
   
    return image_names, mask_names


class Medical_Dataset(Dataset):
    
    def __init__(self, images_path, masks_path, size):
        
        self.images_path = images_path
        self.masks_path = masks_path
        self.height = size[0]
        self.width = size[1]
        self.n_samples = len(images_path)
    
    def __getitem__(self, index):
        
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        
        image1 = cv2.resize(image, (self.width, self.height))
        mask = cv2.resize(mask, (self.width, self.height))
        
        image1 = np.transpose(image1, (2,0,1))
        mask = np.expand_dims(mask, axis = 0)
        
        image1 = image1 / 255.
        mask = mask / 255.
        
        image1 = image1.astype(np.float32)
        mask = mask.astype(np.float32)
        
        image1 = torch.from_numpy(image1)
        mask = torch.from_numpy(mask)
        
        return image1, mask
    
    def __len__(self):
        
        return self.n_samples
