#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import resnet
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
import torch.nn.functional as F
from PIL import Image

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from inspect import signature

from sklearn.metrics import precision_recall_curve
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from inspect import signature
plt.switch_backend('agg')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

# nodules = torch.load('nodules6000:.pt')
# all_lables = torch.load('all_labels6000:.pt')
# print(nodules.shape, all_lables.shape)

# threshold = 3
# all_lables[all_lables < threshold] = 0
# all_lables[all_lables >= threshold] = 1

batch_size = 32
validation_split = .2
shuffle_dataset = True
random_seed= 42


def rotation(img):
    i, j = random.randint(0, 2), random.randint(0, 2)
    return img.transpose(i, j)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

dataset = mt_datasets.TensorDataset(nodules, all_lables, transforms=rotation)
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


model = resnet.resnet18(sample_size=64, sample_duration=64, num_classes=[2])
model.to(device)

#criterion = nn.CrossEntropyLoss()
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


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


        # get the inputs; data is a list of [inputs, labels]
        nodule = nodule.unsqueeze(1)
        nodule = torch.cat([nodule,nodule,nodule],dim=1)
        labels = labels[:, -1]
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(nodule)       

        _, predicted = torch.max(outputs[0].data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()



        loss = 0.0
        for i, output in enumerate(outputs):
            loss += criterion(output, labels)
            
        loss.backward()
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

            # get the inputs; data is a list of [inputs, labels]
            nodule = nodule.unsqueeze(1)
            nodule = torch.cat([nodule,nodule,nodule],dim=1)
            labels = labels[:, -1]

            # forward
            outputs = model(nodule)[0]     

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

#         print('Recall:', recall_score(y_test, y_pred > 0.5), \
#               'Precision:', recall_score(y_test, y_pred > 0.5), \
#               'Precision:', recall_score(y_test, y_pred))
#         print('Confusion matrix\n', confusion_matrix(y_test, y_pred > 0.5))



print('Finished Training')


print('Start Validation')

print("="*20)
print('Precision/Recall Calculation')
print("="*20)


correct = 0
total = 0

y_test, y_scores = [], []
for i, (nodule, labels) in enumerate(validation_loader):
    nodule, labels = nodule[0].to(device), labels.to(device)

    # get the inputs; data is a list of [inputs, labels]
    nodule = nodule.unsqueeze(1)
    nodule = torch.cat([nodule,nodule,nodule],dim=1)
    labels = labels[:, -1]

    # forward
    outputs = model(nodule)[0]     

    y_test.append(labels.detach().cpu())
    m = torch.nn.Softmax(dim=1)
    y_scores.append(m(outputs).detach().cpu())

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    
print(f'Epoch: {epoch}, Accuracy: {(100 * correct / total)}')


y_test = torch.cat(y_test).numpy()
y_scores = torch.cat(y_scores).numpy()
y_pred = y_scores[:,1]

# precision, recall, _ = precision_recall_curve(y_test, y_pred)
# average_precision = average_precision_score(y_test, y_pred)

# print('average precision is', average_precision)
# print('confusion matrix is', confusion_matrix(y_test, y_pred > 0.5))

# # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
# step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
# fig = plt.gcf()
# fig.subplots_adjust(bottom=0.25)
# plt.step(precision, recall,  color='b', alpha=0.2, where='post')
# plt.fill_between(precision, recall, alpha=0.2, color='b', **step_kwargs)

# plt.xlabel('Precision')
# plt.ylabel('Recall')
# plt.ylim([0.0, 1.05])
# plt.xlim([1, 0])
# plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


# plt.show()
# fig.savefig('precesion_recall_4.png')   # save the figure to file
# plt.close(fig)

from sklearn.metrics import roc_curve, auc


fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# # plot no skill
# plt.plot([0, 1], [0, 1], linestyle='--')
# # plot the roc curve for the model
# plt.plot(fpr, tpr, marker='.', label='ROC curve (area = %0.2f)' % roc_auc)
# # show the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
fig.savefig('auc.png')   # save the figure to file
plt.close(fig)
# pyplot.show()
# plt.plot(fpr, tpr, color='darkorange',
#          lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')


