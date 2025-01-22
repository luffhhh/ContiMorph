import torch
import argparse
import glob
import os
import random
import pydicom
import numpy as np
np.bool = bool 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image
from medpy import metric
import nibabel as nib
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision import transforms
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist


def data_augment(image, shift=10.0, rotate=10.0, scale=0.1, intensity=0.1, flip=False):
    # Perform affine transformation on image and label, which are 4D tensors of dimension (N, C, X, Y).
    image2 = np.zeros(image.shape, dtype='float32')
    for i in range(image.shape[0]):
        # Random affine transformation using normal distributions
        shift_var = [np.clip(np.random.normal(), -3, 3) * shift, np.clip(np.random.normal(), -3, 3) * shift]
        rotate_var = np.clip(np.random.normal(), -3, 3) * rotate
        scale_var = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_var = 1 + np.clip(np.random.normal(), -0.5, 0) * intensity

        # Apply affine transformation (rotation + scale + shift) to training images
        row, col = image.shape[2:]
        M = cv2.getRotationMatrix2D((col / 2, row / 2), rotate_var, scale_var)
        M[0, 2] += shift_var[0]
        M[1, 2] += shift_var[1]

        for c in range(image.shape[1]):
            image2[i, c] = ndimage.affine_transform(image[i, c], M[:, :2], offset=M[:, 2], order=1)

        # Apply intensity variation
        if np.random.uniform() >= 0.67:
            image2[i, :] *= intensity_var

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.67:
                image2[i, :] = image2[i, :, ::-1, :]
            elif np.random.uniform() <= 0.33:
                image2[i, :] = image2[i, :, :, ::-1]
                
    return image2

def normalize_data(img_np):
    # preprocessing
    cm = np.median(img_np)
    img_np = img_np / (8*cm)
    img_np[img_np < 0] = 0.0
    img_np[img_np >1.0] = 1.0
    return img_np

def load_nii_to_tensor(file_path):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    normed_data = normalize_data(data)
    return normed_data

def load_nii_to_tensor2(file_path):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    normed_data =  (data - np.min(data)) / (np.max(data) - np.min(data))
    return normed_data

def load_nii_to_tensor255(file_path):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = (data * 255).astype(np.uint8)
    tensor_data = torch.tensor(data)
    
    return tensor_data


def read_info_cfg(file_path):
    info = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if key in ['ED', 'ES', 'NbFrame']:
                info[key] = int(value)
            elif key in ['Height', 'Weight']:
                info[key] = float(value)
            else:
                info[key] = value
    return info

def get_es_value(file_path):
    config_path = os.path.dirname(os.path.dirname(file_path))
    config_path = os.path.join(config_path, 'Info.cfg')
    info = read_info_cfg(config_path)
    return info.get('ED'),info.get('ES'),info.get('Group')

def mask_tensor(file_path):
    ed_value,es_value,group = get_es_value(file_path)
    str_ed_value = "{:02d}".format(ed_value)
    ed_mask_path = file_path.replace('.nii.gz', '_frame'+str_ed_value+'.nii.gz')
    ed_mask = sitk.ReadImage(ed_mask_path)
    ed_mask = sitk.GetArrayFromImage(ed_mask)
    ed_mask = torch.tensor(ed_mask)
    ed_mask = ed_mask.unsqueeze(0)
    str_es_value = "{:02d}".format(es_value)
    es_mask_path = ed_mask_path.replace('_frame01.nii.gz', '_frame'+str_es_value+'.nii.gz')
    es_mask = sitk.ReadImage(es_mask_path)
    es_mask = sitk.GetArrayFromImage(es_mask)
    es_mask = torch.tensor(es_mask)
    es_mask = es_mask.unsqueeze(0)

    return ed_mask, es_mask, es_value,group

class TagDataset(Dataset):
    def __init__(self, data_path, data_type):
        super().__init__()
        self.data_path, self.data_type = data_path, data_type
        fp = open('{}/{}/cine_files.txt'.format(self.data_path,self.data_type),'r')
        imgseqs=[]
        for line in fp:
            line = line.strip('\n')
            line = line.strip()
            if line:
                imgseqs.append(line)
        self.imgseqs = imgseqs
        self.num = len(self.imgseqs)
        self.indices = list(range(self.num))
        if self.data_type == 'train':
            random.shuffle(self.indices)
        
    def __len__(self):
        return self.num

    def __getitem__(self, indx):
        idx = self.indices[indx % self.num]
        seq_path = self.imgseqs[idx]
        cine_imgs = load_nii_to_tensor(seq_path)

        ed_value,es_value,group = get_es_value(seq_path)
        if self.data_type == 'train':
            cine_imgs = cine_imgs + np.random.normal(scale=0.01, size=cine_imgs.shape)
            cine_imgs = data_augment(cine_imgs[np.newaxis], shift=10.0, rotate=10.0, scale=0.1, \
                                                     intensity=0.1, flip=True)
            cine_imgs = np.squeeze(cine_imgs)
            return seq_path, torch.tensor(cine_imgs),es_value
        elif self.data_type == 'test':
            ed_mask,es_mask,es_value,group = mask_tensor(seq_path)
            return seq_path ,torch.tensor(cine_imgs),ed_mask,es_mask,es_value,group
        else:
            return seq_path ,torch.tensor(cine_imgs),es_value
            
def psnr(x, y, data_range=1.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    score = - 10 * torch.log10(mse)
    return score


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=1, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()


def compute_metrics(segmentation, prediction, label):
    binary_gt = (segmentation == label).astype(np.uint8)
    binary_pred= (prediction > 0).astype(np.uint8)
    try:
        hd95 = metric.binary.hd95(binary_gt, binary_pred)
        dice = metric.binary.dc(binary_pred, binary_gt)
    except RuntimeError:
        return np.inf, np.inf
    return hd95, dice


def compute_metrics_all_classes(prediction, segmentation):
    labels = np.unique(segmentation[segmentation > 0]) 
    hd95_dict = {}
    dice_dict = {}
    for label in labels:
        hd95, dice = compute_metrics(segmentation, prediction[int(label)-1], label)
        hd95_dict[label] = hd95
        dice_dict[label] = dice
        
    return hd95_dict, dice_dict

