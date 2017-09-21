This project contains examples of transfer learning using 
convolutional neural networks trained on the Imagenet dataset
and repurposed to identify objects from the [Caltech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)
dataset. Various open-source DL frameworks are used and 
leverage GPU acceleration to perform this task. The example 
here is exactly the same as the exercise described in 
[this blog post](https://blog.cloudera.com/blog/2017/06/deep-learning-on-apache-spark-and-hadoop-with-deeplearning4j/) 
from the Cloudera Engineering blog.

## Download and store source data

The same data is used across all frameworks. Prepare this data initially by 
executing the code in [dataprep.py](dataprep.py).

## Featurize images

For any particular framework, run the code in the `featurize.py` file, but be
sure to change the number of GPUs used depending on your setup.

## Train network

Execute the code in `train.py` to load the featurized images and train a classifier
on the features created in the previous step.
By default, `train.py` simply trains a softmax layer on top of the VGG16 classifier,
which outputs 257 probabilities. Changing the architecture can help eek out performance:
adding BatchNorm, training deeper layers, adding dropout, etc.

## Reference

Some of the code was adapted from the following examples:

### MXNet

* [Gluon image classification example](https://github.com/apache/incubator-mxnet/blob/master/example/gluon/image_classification.py)
* [Gluon documentation](https://gluon.mxnet.io)

### Keras

* [Keras make_parallel script](https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py)
* [fast.ai deep learning course notebook](https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson2.ipynb)

### PyTorch

* [PyTorch tutorial on transfer learning](http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py)

