%cd mxnet-gluon

import mxnet as mx
import os
import random
import time
import numpy as np
from PIL import Image
import mxnet.gluon as gluon
import cv2
from urllib.request import urlopen
import warnings
warnings.filterwarnings('ignore')
from im2rec import read_list

data_dir = "/home/cdsw/train_data/"

def get_iterator(batch_size, path, data_shape=(3, 224, 224)):
  iterator = mx.io.ImageRecordIter(
        path_imgrec         = path,
        data_name           = 'data',
        label_name          = 'label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = False,
        rand_crop           = False,
        rand_mirror         = False,
        mean_r              = 123.68,
        mean_g              = 116.779,
        mean_b              = 103.939,
        scale               = 1 / 255.)
  return iterator


num_classes = 257
batch_per_gpu = 32
num_gpus = 2
batch_size = batch_per_gpu * num_gpus
devs = [mx.gpu(i) for i in range(num_gpus)]
phases = ['train', 'valid', 'test']
path_names = ['/home/cdsw/mxnet-gluon/data/caltech-%s.rec' % phase for phase in phases]
iterators = {phase: get_iterator(batch_size, path) for phase, path in zip(phases, path_names)}

vgg16 = gluon.model_zoo.vision.vgg16(pretrained=True, ctx=devs)
vgg16_feat = vgg16.features
vgg16_feat.hybridize()

def featurize(file_name, iterator):
  '''
  Pass images through the features part of the VGG network, then save them as 
  bytes in a RecordIO file.
  '''
  records = mx.recordio.MXIndexedRecordIO(file_name + '.idx', file_name + '.rec', 'w')
  t0 = time.time()
  batches = []
  k = 0
  for batch in iterator:
    # split minibatch across multiple GPUs
    datas = gluon.utils.split_and_load(batch.data[0], devs, even_split=False)
    labels = gluon.utils.split_and_load(batch.label[0], devs, even_split=False)
    featurized = [(label, vgg16_feat.forward(data)) for label, data in zip(labels, datas)]
    for lbl, out in featurized:
      ln, dn = lbl.asnumpy(), out.reshape((lbl.shape[0], -1)).asnumpy()
      for j in range(ln.shape[0]):
        header = mx.recordio.IRHeader(0, ln[j], k, 0)
        packed = mx.recordio.pack(header, dn[j, :].tobytes())
        records.write_idx(k, packed)
        k += 1
  records.close()
   
  print("Featurized %d images in %0.1f seconds." % (k, time.time() - t0))

featurize('data/valid_feat', iterators['valid'])
featurize('data/train_feat', iterators['train'])
featurize('data/test_feat', iterators['test'])


