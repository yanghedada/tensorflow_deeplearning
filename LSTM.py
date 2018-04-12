# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 00:08:27 2018

@author: yanghe
"""

import tensorflow as tf
import numpy as np

from  tensorflow.examples.tutorials.mnist import  input_data
#mnist = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)

lr = 0.001
train_iters = 100000
batch_size = 100


n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

weights = {'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
           'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
           }

biases = {'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
          'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
          }
def rnn(x,weights,biases):
    x = tf.reshape(x,[-1,n_inputs])
    x_in = tf.matmul(x,weights['in']) + biases['in']
    x_in = tf.reshape(x_in,[-1,n_steps,n_hidden_units])
    
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    cell_init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    output,states = tf.nn.dynamic_rnn(lstm_cell,x_in,initial_state=cell_init_state,time_major=False)
    
    results = tf.matmul(states[1],weights['out']) + biases['out']
    return output,results
output,pred  = rnn(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init  = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #writer = tf.summary.FileWriter("logs", sess.graph)
#==============================================================================
#     step  = 0
#     while step * batch_size < train_iters:
#         batch_xs,batch_ys = mnist.train.next_batch(batch_size)
#         batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
#         sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys})
#         if step % 10 ==0:
#             print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
#         step  += 1 
#==============================================================================














