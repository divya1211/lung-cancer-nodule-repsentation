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

from utils import find


ROOT_DIR = "/scratch/dj1311/Lung_cancer/LIDC_conversion5/"

img_list = []
label_list = []
my_list = []


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
            b = gt.split(".nrrd.nii.gz")[0].split("Annotation")[1]
            s = img.split("LIDC-IDRI-")[1][0:4]
            my_list.append((b, s))

filename_pairs = list(zip(img_list, label_list))
train_dataset = mt_datasets.MRI2DSegmentationDataset(filename_pairs, transform=mt_transforms.ToTensor())


C = 32
nodules = []
malignancies = []

f = open('dev', 'w')

for i, patient in enumerate(train_dataset):
    print(f"Processing patient {i}.")

    if patient is None:
        print(f"Skipping {i}")
        continue
    gt = patient['gt']
    inp = patient['input']
    malignancy = find(*my_list[i])
    if not malignancy:
        print(f"Skipping {i} without malignancy.")
        continue
    malignancies.append(torch.tensor([malignancy]))

    depth = inp.shape[0]

    idx = gt.nonzero()
    min_, max_ =  idx.min(dim=0)[0], idx.max(dim=0)[0]
    start, end = (min_ +  max_)/2 - C, (min_ +  max_)/2 + C
    
    s_x, s_y, s_z = start.tolist()
    e_x, e_y, e_z = end.tolist()

    if s_x < 0:
        e_x += abs(s_x)
        s_x = 0

    if e_x > depth:
        s_x -= (e_x - depth)
        e_x = depth


    if s_y < 0:
        e_y += abs(s_y)
        s_y = 0

    if e_y > 512:
        s_y -= (e_y - depth)
        e_y = depth


    if s_z < 0:
        e_z += abs(s_z)
        s_z = 0

    if e_z > 512:
        s_z -= (e_z - depth)
        e_z = depth


    nodule = (gt * inp)[s_x:e_x, s_y:e_y, s_z:e_z]
    print(i, nodule.shape, file=f)
    f.flush()
    nodules.append(nodule.unsqueeze(0))


nodules = torch.cat(nodules)
malignancies = torch.cat(malignancies)

print(nodules.shape, malignancies.shape)

torch.save(nodules, 'nodules.pt')
torch.save(malignancies, 'malignancies.pt')



