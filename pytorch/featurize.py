%cd pytorch

from __future__ import print_function, division

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

plt.ion()

model_conv = torchvision.models.vgg16(pretrained=True)

class MyScale(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

data_dir = "/home/cdsw/train_data/256_ObjectCategories/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([MyScale(size=[224, 224]), transforms.ToTensor()])
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform)
         for x in ['train', 'test', 'valid']}
batch_size = 32
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size, shuffle=False, num_workers=4)
                for x in ['train', 'test', 'valid']}
dset_classes = dsets['train'].classes
next(iter(dset_loaders['valid']))

class VGGFeaturize(nn.Module):
    def __init__(self):
        super(VGGFeaturize, self).__init__()
        self.features = model_conv.features
        self.classifier = model_conv.classifier
    def forward(self, x):
        x = self.features(x)
        return x.resize(x.size()[0], 25088)
      
model = torch.nn.DataParallel(VGGFeaturize()).cuda()

def featurize(dataset):
    feature_batches = []
    label_batches = []
    for data in dataset:
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        output = model(inputs)
        feature_batches.extend(output.data.cpu().numpy())
        label_batches.extend(labels.data.cpu().numpy())
    features_stacked = np.concatenate([[feat] for feat in feature_batches])
    return features_stacked, label_batches
  
def featurize_and_save(data_label, base_dir):
    t0 = time.time()
    feat, label = featurize(dset_loaders[data_label])
    c_feat = bcolz.carray(feat, rootdir=save_path + 'conv_%s_feat.dat' % data_label)
    c_feat.flush()
    c_label = bcolz.carray(label, rootdir=save_path + 'conv_%s_label.dat' % data_label)
    c_label.flush()
    t1 = time.time()
    print("Saved %s images for %s phase in %0.1f seconds." % (c_feat.shape[0], data_label, (t1 - t0)))

save_path = 'data/'
featurize_and_save('valid', save_path)
featurize_and_save('train', save_path)
featurize_and_save('test', save_path)