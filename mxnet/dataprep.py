%pushd /home/cdsw/mxnet/data
import os
import random

!curl -L -O http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
!tar -xf 256_ObjectCategories.tar
labels = os.listdir("256_ObjectCategories")
!mkdir -p 256_ObjectCategories/train 256_ObjectCategories/test 256_ObjectCategories/valid

for label in labels:
  samples = os.listdir("256_ObjectCategories/" + label)
  random.shuffle(samples)
  ntrain = int(len(samples) * 0.55)
  ntest = int(len(samples) * 0.25)
  os.mkdir("256_ObjectCategories/train/" + label)
  os.mkdir("256_ObjectCategories/test/" + label)
  os.mkdir("256_ObjectCategories/valid/" + label)
  for sample in samples[:ntrain]:
    os.rename("256_ObjectCategories/" + label + "/" + sample, "256_ObjectCategories/train/" + label + "/" + sample)
  for sample in samples[ntrain:ntrain + ntest]:
    os.rename("256_ObjectCategories/" + label + "/" + sample, "256_ObjectCategories/test/" + label + "/" + sample)
  for sample in samples[ntrain + ntest:]:
    os.rename("256_ObjectCategories/" + label + "/" + sample, "256_ObjectCategories/valid/" + label + "/" + sample)  
  os.rmdir("256_ObjectCategories/" + label)

!python3 ../im2rec.py --list True --recursive True mxtrain 256_ObjectCategories/train/
!python3 ../im2rec.py --list True --recursive True mxvalid 256_ObjectCategories/valid/
!python3 ../im2rec.py --newsize "224,224" --quality 100 --num-thread 16 mxvalid 256_ObjectCategories/valid/
!python3 ../im2rec.py --newsize "224,224" --quality 100 --num-thread 16 mxtrain 256_ObjectCategories/train/
%popd