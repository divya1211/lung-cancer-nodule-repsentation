import numpy as np
import os
import datasets as mt_datasets
import transforms as mt_transforms

import torch
# from torchvision import transforms
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
        key = None
        values = []

        for f in filenames:            
            if f.endswith(".nii.gz"):
                if f.startswith("LIDC-IDRI"):
                    key = f
                else:
                    values.append(f)
            
        for value in values:
            img_list.append(os.path.join(dirpath, key))
            label_list.append(os.path.join(dirpath, value))
            # print(img_list[-1], label_list[-1])
            # sys.exit()
            patient_id = key.split("LIDC-IDRI-")[1][0:4]
            nodule_id = value.split(".nrrd.nii.gz")[0].split("Annotation")[1]
            my_list.append((patient_id, nodule_id))


old = 450+451+452+453+454+455+456#+457+458+459+460
num = 457
img_list, label_list = img_list[old:old+num], label_list[old:old+num]
filename_pairs = list(zip(img_list, label_list))
train_dataset = mt_datasets.MRI2DSegmentationDataset(filename_pairs, transform=mt_transforms.ToTensor())


C = 45
nodules = []
all_labels = []

f = open('dev', 'w')

for i, patient in enumerate(train_dataset):
    print(f"Processing patient {i}.")

    if patient is None:
        print(f"Skipping {i}:")
        continue
    gt = patient['gt']
    inp = patient['input']
    labels = find(*my_list[i])
    if not labels:
        print(f"Skipping {i}: { patient['filename']}, {patient['gt_filename']}, without lables.")
        continue

    idx = gt.nonzero()
    min_, max_ =  idx.min(dim=0)[0], idx.max(dim=0)[0]
    start, end = (min_ +  max_)/2 - C, (min_ +  max_)/2 + C
    
    s_x, s_y, s_z = start.tolist()
    e_x, e_y, e_z = end.tolist()

    x, y, z = inp.shape
    
    
    if s_x < 0:
        s_x = 0
        e_x = 90

    if s_y < 0:
        s_y = 0
        e_y = 90

    if s_z < 0:
        s_z = 0
        e_z = 90

    if e_z > z:
        s_z = z-90
        e_z = z

    if e_y > y:
        s_y = y-90
        e_y = y

    if e_x > x:
        s_x = x-90
        e_x = x
    


    nodule = (inp)[s_x:e_x, s_y:e_y, s_z:e_z]
    
    print(i, nodule.shape, patient['filename'], patient['gt_filename'], file=f)
    
    f.flush()

    if nodule.shape[0]== 90 and nodule.shape[1]== 90 and nodule.shape[2]== 90:
        nodules.append(nodule.unsqueeze(0))
        all_labels.append(torch.tensor(labels).unsqueeze(0))
    else:
        print(i,nodule.shape)



nodules = torch.cat(nodules)
all_labels = torch.cat(all_labels, dim=0)

print(nodules.shape, all_labels.shape)

torch.save(nodules, f'nodules{num}:.pt')
torch.save(all_labels, f'all_labels{num}:.pt')



