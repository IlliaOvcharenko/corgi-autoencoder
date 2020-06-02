import torch
import torchvision
import cv2

import numpy as np
import pandas as pd
import albumentations as A


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def denormalize(img_tensor):
    img_tensor = img_tensor.clone()
    for t, m, s in zip(img_tensor, MEAN, STD):
        t.mul_(s).add_(m)
    return img_tensor


def image_to_std_tensor(image, **params):
    image = torchvision.transforms.functional.to_tensor(image)
    image = torchvision.transforms.functional.normalize(image, MEAN, STD)
    return image


def mask_to_tensor(mask, **params):
    return torch.tensor(mask).float()

custom_to_std_tensor = A.Lambda(image=image_to_std_tensor, mask=mask_to_tensor)

def rgb_to_lab(image, **params):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

def lab_to_rgb(image, **params):
    return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

rgb_to_lab_transform = A.Lambda(image=rgb_to_lab)
lab_to_rgb_transform = A.Lambda(image=lab_to_rgb)

def image_to_array(image, **params):
#     return np.transpose((image * 255).numpy(), (1, 2, 0)).astype(np.uint8)
    return np.array(torchvision.transforms.functional.to_pil_image(image))