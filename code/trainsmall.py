import random
import numpy as np

import resnet
import datasets as mt_datasets

import torch
from torchvision import transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict
import torch.nn.functional as F

from utils import FocalLoss, rotation


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

nodules = torch.load('concat_nodules_final.pt')
all_lables = torch.load('concat_lables_final.pt')
print(nodules.shape, all_lables.shape)

threshold = 3
all_lables[all_lables < threshold] = 0
all_lables[all_lables >= threshold] = 1

batch_size = 64
validation_split = .2
shuffle_dataset = True
random_seed= 42
torch.manual_seed(random_seed)

#filename_pairs = get_filename_pairs()
#dataset = mt_datasets.MRI2DSegmentationDataset(filename_pairs)

dataset = mt_datasets.TensorDataset(nodules, all_lables)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)


model = resnet.resnet18(sample_size=90, sample_duration=30, num_classes=2)
model.to(device)

#criterion = nn.CrossEntropyLoss()
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)


print("="*20)
print('Training Started')
print("="*20)


epochs = 100
idx = 0
y_true = []
y_scores = []
for epoch in range(epochs):
    g_loss = 0
    correct = 0
    total = 0
    for i, (nodule, labels) in enumerate(train_loader):
        nodule, labels = nodule.to(device), labels.to(device)
        #labels = labels.squeeze()
        labels = labels.view(-1)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        #print("Forward pass")
        # forward + backward + optimize
        output = model(nodule)
        #print(f"output: {output.shape} {torch.max(output)}")

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # loss = 0.0
        # for i, output in enumerate(output):
        loss = criterion(output, labels)
        # print(f"loss: {loss}")
        
        #print("Backward pass")
        loss.backward()
        #print("Backward pass done.")

        g_loss += loss.item()
        optimizer.step()

        # print statistics
        writer.add_scalar('Loss/train', loss.item(), idx)
        idx += 1

    print(f"Epochs: {epoch}, i = {i}, Loss = {g_loss}, Train Accuracy: {(100 * correct / total)}")

    if epoch % 2 == 0:
        correct = 0
        total = 0

        y_test, y_scores = [], []
        for i, (nodule, labels) in enumerate(validation_loader):
            nodule, labels = nodule.to(device), labels.to(device)
            #labels = labels.squeeze()
            labels = labels.view(-1)

            # forward
            outputs = model(nodule)

            y_test.append(labels.detach().cpu())
            m = torch.nn.Softmax(dim=1)
            y_scores.append(m(outputs).detach().cpu())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        print(f'Epoch: {epoch}, Validaiton Accuracy: {(100 * correct / total)}')
        y_test = torch.cat(y_test).numpy()
        y_scores = torch.cat(y_scores).numpy()
        y_pred = y_scores[:,1]

