%pushd /home/cdsw/train_data
import os
import random

source_tar = "/home/cdsw/source_data/256_ObjectCategories.tar"
!tar -xf $source_tar
labels = os.listdir("256_ObjectCategories")
!mkdir -p 256_ObjectCategories/train 256_ObjectCategories/test 256_ObjectCategories/valid

train_split = 0.55
test_split = 0.25
valid_split = 1.0 - train_split - test_split
for label in labels:
  samples = os.listdir("256_ObjectCategories/" + label)
  random.shuffle(samples)
  ntrain = int(len(samples) * train_split)
  ntest = int(len(samples) * test_split)
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
%popd