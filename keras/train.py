 from __future__ import division,print_function

import os
import numpy as np
import tensorflow as tf
import bcolz
import time

from keras import backend as K
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Model
import keras
from keras.optimizers import Adam
from keras.preprocessing import image

path = "./keras/data/"
conv_feat = bcolz.open(path + 'conv_feat.dat')
conv_val_feat = bcolz.open(path + 'conv_val_feat.dat')
conv_test_feat = bcolz.open(path + 'conv_test_feat.dat')

feat_input = Input(shape=(7, 7, 512))
y = Flatten(name='flatten')(feat_input)
y = Dense(4096, activation='relu', name='fc1')(y)
y = Dense(4096, activation='relu', name='fc2')(y)
y = Dropout(rate=0.2)(y)
y = Dense(257, activation='softmax', name='predictions')(y)

model_top = Model(feat_input, y, name='vgg16_top')

for layer_name in ['fc1', 'fc2']:
    model_top.get_layer(layer_name).set_weights(model_full.get_layer(layer_name).get_weights())
    
for layer_name in ['fc1', 'fc2']:
    model_top.get_layer(layer_name).trainable = False
from keras.utils.np_utils import to_categorical
generator = image.ImageDataGenerator()
path = 'data/256_ObjectCategories/'
batches = generator.flow_from_directory(path + 'train', target_size=(224, 224), class_mode='categorical', shuffle=False, batch_size=batch_size)
val_batches = generator.flow_from_directory(path + 'valid', target_size=(224, 224), class_mode='categorical', shuffle=False, batch_size=batch_size)
test_batches = generator.flow_from_directory(path + 'test', target_size=(224, 224), class_mode='categorical', shuffle=False, batch_size=batch_size)
(val_classes, trn_classes, val_labels, trn_labels) = \
(val_batches.classes, batches.classes, to_categorical(val_batches.classes), to_categorical(batches.classes))

model_top.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model_top.fit(conv_feat, trn_labels, batch_size=batch_size, epochs=5, validation_data=(conv_val_feat, val_labels))