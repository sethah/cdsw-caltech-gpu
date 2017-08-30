import mxnet as mx
import os
import random
from mxnet.image import ImageIter
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
%matplotlib inline
%cd mxnet

import warnings
warnings.filterwarnings('ignore')
from im2rec import read_list

def get_iterators(batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec         = '/home/cdsw/mxnet/data/mxtrain.rec',
        data_name           = 'data',
        label_name          = 'fc7_output',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = False,
        rand_crop           = False,
        rand_mirror         = False)
    val = mx.io.ImageRecordIter(
        path_imgrec         = '/home/cdsw/mxnet/data/mxvalid.rec',
        data_name           = 'data',
        label_name          = 'fc7_output',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = False,
        rand_crop           = False,
        rand_mirror         = False)
    return (train, val)
num_classes = 257
batch_per_gpu = 32
num_gpus = 2
batch_size = batch_per_gpu * num_gpus
devs = [mx.gpu(i) for i in range(num_gpus)]
(train, val) = get_iterators(batch_size)

import os, urllib.request
def download(url, filename):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

def get_model(prefix, epoch):
    download(prefix+'-symbol.json', './vgg16-symbol.json')
    download(prefix+'-%04d.params' % (epoch,), './vgg16-%04d.params' % (epoch,))

get_model('http://data.mxnet.io/models/imagenet/vgg/vgg16', 0)


sym, arg_params, aux_params = mx.model.load_checkpoint('vgg16', 0)

# a module contains a network, input and output names, and a context
net = sym.get_internals()['flatten_0_output']
new_params = {k: v for k, v in arg_params.items() if 'fc' not in k}
mod = mx.mod.Module(symbol=net, context=devs, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (batch_size, 3, 224, 224))], label_shapes=mod._label_shapes)
mod.set_params(new_params, aux_params, allow_missing=True)

p, i, b = next(mod.iter_predict(train, reset=False))

def featurize_records(data_iter, file_name):
  """
  Predict in batches and write them to .rec files
  """
  records = mx.recordio.MXIndexedRecordIO(file_name + '.idx', file_name + '.rec', 'w')
  k = 0
  for preds, i, batch in mod.iter_predict(data_iter, reset=False):
    batch_labels = batch.label[0].asnumpy()
    p = preds[0].asnumpy()
    for j in range(p.shape[0]):
      header = mx.recordio.IRHeader(0, batch_labels[j], i, 0)
      packed = mx.recordio.pack(header, p[j, :].tobytes())
      records.write_idx(k, packed)
      k += 1
    if i % 100 == 0:
      print('iter %s' % i)
  records.close()

t0 = time.time()
featurize_records(train, './data/train_feat')
featurize_records(val, './data/val_feat')
t1 = time.time()
print(t1 - t0)