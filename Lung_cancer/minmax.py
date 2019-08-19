import numpy as np
import os
import datasets as mt_datasets
import transforms as mt_transforms

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

from collections import defaultdict
from pathlib import Path
import imageio

ROOT_DIR = "/scratch/dj1311/Lung_cancer/LIDC_conversion5/"

img_list = []
label_list = []

for idx, (dirpath, dirnames, filenames) in enumerate(os.walk(ROOT_DIR)):
    if not dirnames:
        # for 1 patient, all files
        nii = []
        flag = False
        for f in filenames:
            if f.startswith("LIDC-IDRI") and f.endswith(".nii.gz"):
                img = f
                nii.append(f)
                    
            if flag == False and f.startswith("Nodule 1 - Annotation") and f.endswith(".nii.gz"):
                gt = f
                nii.append(f)
                flag = True
            
        if len(nii) == 2:
            img_list.append(os.path.join(dirpath, img))
            label_list.append(os.path.join(dirpath, gt))

filename_pairs = list(zip(img_list, label_list))
train_dataset = mt_datasets.MRI2DSegmentationDataset(filename_pairs, transform=mt_transforms.ToTensor())

global_max = []

for patient in train_dataset:
    inp, gt = patient['input'], patient['gt']
    
    idx = gt.nonzero()
    min_, max_ =  idx.min(dim=0)[0], idx.max(dim=0)[0]
    global_max.append((max_ - min_).tolist())



a, b, c = max([x[0] for x in global_max]), max([x[1] for x in global_max]), max([x[2] for x in global_max])
print(a, b, c)

