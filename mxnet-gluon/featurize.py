%cd mxnet-gluon

import mxnet as mx
import os
import random
import time
import numpy as np
from PIL import Image
import mxnet.gluon as gluon
from mxnet.ndarray._internal import _cvimresize as imresize
import cv2
from urllib.request import urlopen
import warnings
warnings.filterwarnings('ignore')

num_classes = 257
batch_per_gpu = 32
num_gpus = 2
batch_size = batch_per_gpu * num_gpus
devs = [mx.gpu(i) for i in range(num_gpus)]


def image_transform(data, label, height=224, width=224):
  new_image = mx.nd.transpose(imresize(data, height, width), (2, 0, 1)).astype(np.float32)
  new_image /= 255.
  return new_image, label

data_dir = "/home/cdsw/train_data/"
phases = ['train', 'test', 'valid']
datasets = {phase: gluon.data.vision.ImageFolderDataset(data_dir + '256_ObjectCategories/' + phase, flag=1,
                                               transform=image_transform) for phase in phases}
loaders = {phase: gluon.data.DataLoader(datasets[phase], batch_size=batch_size, shuffle=False) for phase in phases}

vgg16 = gluon.model_zoo.vision.vgg16(pretrained=True, ctx=devs)
vgg16_feat = vgg16.features
vgg16_feat.hybridize()

def featurize(file_name, iterator):
  batches = []
  for data, label in iterator:
    datas = gluon.utils.split_and_load(data, devs, even_split=False)
    labels = gluon.utils.split_and_load(label, devs, even_split=False)
    featurized = [(label, vgg16_feat.forward(data)) for label, data in zip(labels, datas)]
    batches.extend(featurized)
   
  records = mx.recordio.MXIndexedRecordIO(file_name + '.idx', file_name + '.rec', 'w')
  k = 0
  for l, d in batches:
    ln, dn = l.asnumpy(), d.reshape((l.shape[0], -1)).asnumpy()
    for j in range(ln.shape[0]):
      header = mx.recordio.IRHeader(0, ln[j], k, 0)
      packed = mx.recordio.pack(header, dn[j, :].tobytes())
      records.write_idx(k, packed)
      k += 1
  records.close()

t0 = time.time()
featurize('data/train_feat', loaders['train'])
t1 = time.time()
featurize('data/valid_feat', loaders['valid'])
t2 = time.time()
featurize('data/test_feat', loaders['test'])
t3 = time.time()
print("Train featurize step took %0.1f seconds." % (t1 - t0))
print("Valid featurize step took %0.1f seconds." % (t2 - t1))
print("Test featurize step took %0.1f seconds." % (t3 - t2))


