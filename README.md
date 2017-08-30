This project contains examples of transfer learning using 
convolutional neural networks trained on the Imagenet dataset
and repurposed to identify objects from the [Caltech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)
dataset. Various open-source DL frameworks are used and 
leverage GPU acceleration to perform this task. The example 
here is exactly the same as the exercise described in 
[this blog post](https://blog.cloudera.com/blog/2017/06/deep-learning-on-apache-spark-and-hadoop-with-deeplearning4j/) 
from the Cloudera Engineering blog.

# Setup

## MXNet

````
pip3 install mxnet-cu80
pip3 install opencv-python
pushd ~/mxnet
curl -o im2rec.py https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py
popd
````

**Note**: Had to make a couple changes to im2rec.py unfortunately.

## Tensorflow

````
pip3 install tensorflow-gpu
````

## Keras

````
pip3 install tensorflow-gpu
pip3 install keras
pip3 install bcolz
pip3 install h5py
pip3 install pillow
````

## PyTorch

````
pip3 install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp36-cp36m-linux_x86_64.whl 
pip3 install torchvision
````


