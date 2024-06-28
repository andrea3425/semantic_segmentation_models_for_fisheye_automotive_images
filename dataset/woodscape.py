"""
Implementation of the WoodScapesDataset class 
"""

import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from edge_utils import generate_edgemap

class WoodScapesDataset(Dataset):
    """ This class is responsible for loading images and their corresponding ground truth semantic masks from the dataset directory."""
    def __init__(self, img_dir, annotation_dir, num_classes=9, idxs=None, transform=None, target_transform=None, joint_transform=None, include_edgemap=False):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform=joint_transform
        self.idxs = idxs
        self.num_classes = num_classes
        self.include_edgemap = include_edgemap
        self.images = sorted([os.path.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith('.png')])
        if self.idxs:
            self.images = sorted([self.images[idx] for idx in range(len(self.images)) if idx in self.idxs])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        annotation_path = os.path.join(self.annotation_dir, os.path.basename(img_path))
        image = Image.open(img_path).convert("RGB")
        annotation = Image.open(annotation_path)

        # random scaling in the range of [0.5, 2] and random crop of 483×483 image patches
        if self.joint_transform:
            image, annotation = self.joint_transform(image, annotation)

        # image transforms
        if self.transform:
            image = self.transform(image)

        # mask transforms
        if self.target_transform:
            annotation = self.target_transform(annotation)
        
        if self.include_edgemap:
            edgemap = generate_edgemap(annotation)
            return image, annotation, edgemap
        
        return image, annotation

    def data(self, idx):
        return Image.open(self.images[idx]).convert("RGB")
