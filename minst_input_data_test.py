# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 09:40:47 2018

@author: yanghe
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

data = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)
#==============================================================================
# mnist.train
# mnist.test
# mnist.validation
# mnist.train.images
# mnist.train.labels
#==============================================================================
print(data.train.images.shape)
print(data.train.labels.shape)
print(data.test.images.shape)
print(data.test.labels.shape)
print(data.validation.images.shape)
print(data.validation.labels.shape)