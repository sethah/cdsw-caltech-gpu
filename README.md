This project contains examples of transfer learning using 
convolutional neural networks trained on the Imagenet dataset
and repurposed to identify objects from the [Caltech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)
dataset. Various open-source DL frameworks are used and 
leverage GPU acceleration to perform this task. The example 
here is exactly the same as the exercise described in 
[this blog post](https://blog.cloudera.com/blog/2017/06/deep-learning-on-apache-spark-and-hadoop-with-deeplearning4j/) 
from the Cloudera Engineering blog.

## Data

The same data is used across all frameworks. Prepare this data initially by 
executing the code in dataprep.py.

