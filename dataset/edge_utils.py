"""
Implementation of edge map generator functions:

These functions are designed to produce edge maps, which are annotations marking the semantic boundaries 
within the images. These functions have been adapted from those used in the experiments conducted in 
(Takikawa 2019), available at (https://github.com/nv-tlabs/GSCNN/blob/master/datasets/edge_utils.py).

These functions are utilized within the __getitem__ method of the WoodScapesDataset class to generate edge maps. 
During training, this method returns a tuple containing (image, mask, edgemap). While, during the validation 
phase, the edge map is not generated.
"""

import os
import numpy as np
import torch
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt

def generate_edgemap(mask):
    edgemap = mask.numpy()
    edgemap = mask_to_onehot(edgemap, 9)
    edgemap = onehot_to_binary_edges(edgemap, 2, 9)
    edgemap = torch.from_numpy(edgemap).float()
    return edgemap

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == (i + 1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot_to_mask(mask):
    """
    Converts a mask (K,H,W) to (H,W)
    """
    _mask = np.zeros_like(mask)
    _mask[1:] = mask[:-1]
    _mask = np.argmax(_mask, axis=0)[:-1]
    #_mask[_mask != 0] += 1
    return _mask

def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)

    return np.array(channels)

def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap