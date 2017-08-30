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

import os, urllib.request
def download(url, filename):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

def get_model(prefix, epoch):
    download(prefix+'-symbol.json', './resnet-101-symbol.json')
    download(prefix+'-%04d.params' % (epoch,), './resnet-101-%04d.params' % (epoch,))

get_model('http://data.mxnet.io/models/imagenet/resnet/101-layers/resnet-101', 0)
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet101', 0)
