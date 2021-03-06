%cd mxnet-gluon

import mxnet as mx
import os
import random
from mxnet.image import ImageIter
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import mxnet.gluon as gluon
import warnings
warnings.filterwarnings('ignore')

class ArrayRecordDataset(gluon.data.dataset.RecordFileDataset):
    '''
    A simple dataset that loades records from an IndexedRecordIO file
    and converts them into a numpy array.
    '''
    def __init__(self, filename, array_shape):
        super(ArrayRecordDataset, self).__init__(filename)
        self.data_shape = array_shape

    def __getitem__(self, idx):
        record = super(ArrayRecordDataset, self).__getitem__(idx)
        header, _bytes = mx.recordio.unpack(record)
        arr = np.frombuffer(_bytes, dtype=np.float32).reshape(self.data_shape)
        return arr, header.label
      
phases = ['train', 'valid']
batch_size_per_gpu = 32
num_gpus = 2
batch_size = batch_size_per_gpu * num_gpus
datasets = {phase: ArrayRecordDataset('data/%s_feat.rec' % phase, (25088)) for phase in phases}
loaders = {phase: gluon.data.DataLoader(datasets[phase], batch_size=batch_size,
                                        shuffle=(phase == 'train')) for phase in phases}
  
devs = [mx.gpu(i) for i in range(num_gpus)]
vgg16 = gluon.model_zoo.vision.vgg16(pretrained=True, ctx=devs)
vgg16_cls = vgg16.classifier
with vgg16_cls.name_scope():
  vgg16_cls._children.pop()
  vgg16_cls.register_child(gluon.nn.Dense(257, prefix="predictions_"))
  
# get a parameter dict of the frozen and non-frozen parameters
trainable_params = gluon.parameter.ParameterDict()
fixed_params = gluon.parameter.ParameterDict()
for layer in vgg16_cls._children:
  if 'predictions' in layer.name:
    trainable_params.update(layer.collect_params())
  else:
    fixed_params.update(layer.collect_params())
fixed_params.setattr('grad_req', 'null')  # do not do the backward pass on these params

loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
  
def train_batch(data, label, ctx, net, trainer):
    # split the data batch and load them on GPUs
    data = gluon.utils.split_and_load(data, ctx)
    label = gluon.utils.split_and_load(label, ctx)
    # compute gradient
    forward_backward(net, data, label, ctx)
    # update parameters
    trainer.step(np.sum([d.shape[0] for d in data]))

def valid_batch(data, label, ctx, net):
    data = data.as_in_context(ctx[0])
    pred = mx.nd.argmax(net(data), axis=1)
    return mx.nd.sum(pred == label.astype(np.float32).as_in_context(ctx[0])).asscalar()
  
def forward_backward(net, data, label, ctx):
    with gluon.autograd.record():
        outs = [net(X) for X in data]
        losses = [loss_function(out, Y) for out, Y in zip(outs, label)]
    for l in losses:
        l.backward()
        
def run(num_gpus, lr, num_epochs):
    # the list of GPUs that will be used
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('Running on {}'.format(ctx))

    train_data = loaders['train']
    valid_data = loaders['valid']
    print('Batch size is {}'.format(train_data._batch_sampler._batch_size))
    
    trainable_params.initialize(ctx=ctx, force_reinit=True)

    trainer = gluon.Trainer(trainable_params, 'adam', {'learning_rate': lr})
    for epoch in range(num_epochs):
        start = time.time()
        num_samples = 0
        for data, label in train_data:
            num_samples += data.shape[0]
            train_batch(data, label, ctx, vgg16_cls, trainer)
        mx.nd.waitall()  # wait until all computations are finished to benchmark the time
        print('Epoch %d, samples=%d, training time = %.1f sec' % 
              (epoch, num_samples, time.time() - start))
        # validation set
        correct, num = 0.0, 0.0
        for data, label in valid_data:
            correct += valid_batch(data, label, ctx, vgg16_cls)
            num += data.shape[0]
        print('         validation accuracy = %.4f' % (correct / num))
    
    correct, num = 0.0, 0.0
    for data, label in train_data:
        correct += valid_batch(data, label, ctx, vgg16_cls)
        num += data.shape[0]
    print('         train accuracy = %.4f' %(correct/num), correct, num)
run(num_gpus, 0.001, 5)



