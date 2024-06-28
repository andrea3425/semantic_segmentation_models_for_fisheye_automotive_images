import numpy as np
import torch
import random
import cv2
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image

### Input Image Transformations

cs_mean, cs_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Normalize(cs_mean, cs_std),
])
test_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(size=(960, 1280)),
    #transforms.ColorJitter(brightness=0.2),
    transforms.Normalize(cs_mean, cs_std),
])

### Annotations Transformations

class_id = torch.tensor([0,1,1,2,3,4,5,6,7,8])

def target_import(t):
    t = torch.from_numpy(np.array(t)).long()
    t = class_id[t].long()
    return t

annotation_transform = transforms.Compose([
    transforms.Lambda(lambda t : target_import(t)),
])

test_annotation_transform = transforms.Compose([
    transforms.Lambda(lambda t : target_import(t)),
    transforms.CenterCrop(size=(960, 1280)),
])


### Joint Transforms

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class Rescale(object):
    """Rescale the image in a sample to a given size within a specified range."""
    def __init__(self, scale_range=(0.5, 2.0)):
        assert isinstance(scale_range, (tuple, list)) and len(scale_range) == 2
        self.scale_range = scale_range

    def __call__(self, img, mask):
        scale = random.uniform(*self.scale_range)
        new_size = (int(img.width * scale), int(img.height * scale))

        return img.resize(new_size, Image.BILINEAR), mask.resize(new_size, Image.NEAREST)


class RandomCrop(object):
    """Crop randomly the image in a sample."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img, mask):
        w, h = img.size
        new_h, new_w = self.output_size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        img = img.crop((left, top, left + new_w, top + new_h))
        mask = mask.crop((left, top, left + new_w, top + new_h))

        return img, mask

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and mask randomly with a given probability."""
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img, mask):
        if random.random() < self.probability:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

joint_transform = Compose([
    Rescale(), 
    RandomCrop(483),  
    RandomHorizontalFlip() 
])

joint_transform_2 = Compose([
    Rescale(scale_range=(0.55, 2.0)), 
    RandomCrop(512),  
    RandomHorizontalFlip() 
])

UnNormalize = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                  transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
                                  ])