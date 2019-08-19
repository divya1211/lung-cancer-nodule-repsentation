#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torchvision import transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
import datasets as mt_datasets
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import imageio


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

nodules = torch.load('nodules.pt')
malignancies = torch.load('malignancies.pt') - 1

batch_size = 32
validation_split = .1
shuffle_dataset = True
random_seed= 42


def rotation(img):
    i, j = random.randint(0, 2), random.randint(0, 2)
    return img.transpose(i, j)



dataset = mt_datasets.TensorDataset(nodules, malignancies, transforms=rotation)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)


resnet18 = models.resnet18(pretrained=True)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 5)
resnet18.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=0.001)


epochs = 100
idx = 0
for epoch in range(epochs):
    for i, (nodule, malignancy) in enumerate(train_loader, 0):
        nodule, malignancy = nodule.to(device), malignancy.to(device)


        # get the inputs; data is a list of [inputs, labels]
        input = torch.cat([nodule.mean(dim=-1).unsqueeze(1), nodule.mean(dim=-2).unsqueeze(1), nodule.mean(dim=-3).unsqueeze(1)], dim=1)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet18(input)
        loss = criterion(outputs, malignancy)
        loss.backward()
        optimizer.step()

        # print statistics
        writer.add_scalar('Loss/train', loss.item(), idx)
        idx += 1

    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            input = torch.cat([images.mean(dim=-1).unsqueeze(1), images.mean(dim=-2).unsqueeze(1), images.mean(dim=-3).unsqueeze(1)], dim=1)
            outputs = resnet18(input)
            
            # print(len(outputs))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epochs: {epoch}, i = {i}, Loss = {loss.item()}")
    print(f'Epoch: {epoch}, Accuracy: {(100 * correct / total)}')
    

print('Finished Training')


