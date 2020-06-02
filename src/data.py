import torch
import torchvision
import cv2

import numpy as np
import pandas as pd
import albumentations as A

from pathlib import Path


class CorgiDataset(torch.utils.data.Dataset):
    def __init__(self, df, folder, transform=None):
        self.df = df
        self.folder = folder
        self.transform = transform
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        origin_fn = self.folder / (item.shortcode + ".jpg")
        
        origin = cv2.imread(str(origin_fn))
        origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
        
        if self.transfrom is not None:
            transformed = self.transform(image=origin)
            origin = transformed["image"]
            
        return {"shortcode": item.shortcode, "origin": origin}
        
    def __len__(self):
        return len(self.df)