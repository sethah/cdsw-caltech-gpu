from __future__ import print_function, division

%cd caltech-gpu/pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import Image, ImageOps
import collections
import bcolz
import time
import sys

class BcolzDataset(torch.utils.data.Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index].ravel()), int(self.labels[index])

    def __len__(self):
        return self.features.shape[0]
  
save_path = 'data/'
dsets = {x: BcolzDataset(bcolz.open(save_path + 'conv_%s_feat.dat' % x), bcolz.open(save_path + 'conv_%s_label.dat' % x))
         for x in ['train', 'valid']}
batch_size = 32
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                for x in ['train','valid']}

model_conv = torchvision.models.vgg16(pretrained=True)
def train_model(model, criterion, optimizer, num_epochs=5):
    since = time.time()

    best_model = model
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            for data in dset_loaders[phase]:
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model
  
dset_sizes = {x: len(dsets[x]) for x in ['train', 'valid']}
  
layers = []
for i in range(6):
    layer = model_conv.classifier[i]
    layers.append(layer)
model = nn.Sequential(*layers)
for m in model.parameters():
    m.requires_grad = False
model.add_module('predictions', nn.Linear(4096, 257))
model = model.cuda()

optimizer = torch.optim.Adam(model.predictions.parameters(), lr=0.0001)  # only pass in parameters that will be optimized
criterion = nn.CrossEntropyLoss()
best_model = train_model(model, criterion, optimizer, 5)