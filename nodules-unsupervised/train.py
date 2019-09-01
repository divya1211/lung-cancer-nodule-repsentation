import random
import numpy as np
from sklearn.metrics import roc_auc_score
import resnet
from torchvision import transforms
import torchvision
import datasets as mt_datasets

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

nodules = torch.load('nodules.pt')
malignancies = torch.load('all_labels.pt')

batch_size = 32
validation_split = .1
shuffle_dataset = True
random_seed= 42


def rotation(img):
    i, j = random.randint(0, 2), random.randint(0, 2)
    return img.transpose(i, j)



dataset = mt_datasets.TensorDataset(nodules, malignancies)
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


model = resnet.resnet18(sample_size=64, sample_duration=64, num_classes=[5, 5, 6, 5, 5, 5, 5, 5, 5])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 1
idx = 0
y_true = []
y_scores = []

for epoch in range(epochs):
    for i, (nodule, labels) in enumerate(train_loader, 0):
        nodule, labels = nodule[0].to(device), labels.to(device)

        # get the inputs; data is a list of [inputs, labels]
        nodule = nodule.unsqueeze(1)
        nodule = torch.cat([nodule,nodule,nodule],dim=1)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(nodule)
      
        loss = 0.0
        for i, output in enumerate(outputs):
            loss += criterion(output, labels[i])
            
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
            images, labels = images[0].to(device), labels.to(device)
            images = images.unsqueeze(1)
            images = torch.cat([images,images,images],dim=1)
            outputs = model(images)
            
            # print(len(outputs))
            _, predicted = torch.max(outputs.data, 1)
            # y_true.append(labels)
            # y_scores.append(predicted) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epochs: {epoch}, i = {i}, Loss = {loss.item()}")
    print(f'Epoch: {epoch}, Accuracy: {(100 * correct / total)}')
    

print('Finished Training')

y_true = []
y_scores = []
scores =[]
precision = dict()
recall = dict()
average_precision = dict()

for data in validation_loader:
    images, labels = data
    images, labels = images[0].to(device), labels.to(device)
    images = images.unsqueeze(1)
    images = torch.cat([images,images,images],dim=1)
    outputs = model(images)
    
    y_true.append(labels)
    y_scores.append(outputs)



