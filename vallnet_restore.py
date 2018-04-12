# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 18:46:34 2018

@author: Administrator
"""

#!/usr/bin/env python
# load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)
# start tensorflow interactiveSession
import tensorflow as tf

# weight initialization
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

# convolution
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Create the model
# placeholder
#输入正确值,标签值(新的占位符)
#这里的None表示此张量的第一个维度可以是任何长度的

  # Create the model
x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
# variables
# first convolutinal layer#第一层卷积层
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])
#第一层池化层
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer#第二层卷积层
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#第二层池化层
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer#密集链接层
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer#输出层
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# train and evaluate the model#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

#梯度下降算法（AdamOptimizer）以0.01的学习速率最小化交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#train_step = tf.train.AdagradOptimizer(1e-5).minimize(cross_entropy)

#测试真实标签匹配
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#创建计算图
init = tf.global_variables_initializer()
#初始化我们创建的变量

#创建计算图
sess = tf.InteractiveSession()
sess.run(init)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(init)
saver.restore(sess,'saver/moedl1.ckpt')
conv1 = sess.run(h_conv1,feed_dict={x_image: mnist.test.images[:100], keep_prob: 1.0})
print(conv1.shape)
