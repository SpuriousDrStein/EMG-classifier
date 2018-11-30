import tensorflow as tf
import numpy as np
import math as m
import serial as ser
import os
from tensorflow.contrib.nn import conv1d_transpose
from tensorflow.contrib import autograph

class processer:
    
    def __init__(self, z_channels, rnn_hidden_size, depth=5, input_size=30, batch_size=1): # input_size x 1
        self.input_size = input_size
        self.depth = depth
        self.h_size = rnn_hidden_size
        self.batch_size = batch_size

        self.z_shape = (int(input_size/pow(2, depth)) ,z_channels)
        
    def encoder(self, x, k_size=5, pool_size=2, reuse=False):
        with tf.variable_scope('ENCODER'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            nn = [None for _ in range(self.depth)]
            c = [*reversed([int(self.z_shape[1]/pow(2,d)) for d in range(0, self.depth+1)])]
            print('encoder channels: ', c)

            nn[0] = tf.layers.conv1d(x, c[1], k_size, padding='SAME', kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.constant(0), activation=tf.nn.relu)
            nn[0] = tf.layers.max_pooling1d(nn[0], pool_size, 2)
            for d in range(1, self.depth-1):
                nn[d] = tf.layers.conv1d(nn[d-1], c[d+1], k_size, padding='SAME', kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.constant(0), activation=tf.nn.relu)
                nn[d] = tf.layers.max_pooling1d(nn[d], pool_size, 2)
            nn[-1] = tf.layers.conv1d(nn[-2], c[-1], k_size, padding='SAME', kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.constant(0), activation=tf.nn.relu)
            nn[-1] = tf.layers.max_pooling1d(nn[-1], pool_size, 2)
        return nn[-1]

    def fully_connected(self, x, h_layer_sizes=(200,200), reuse=False):
        with tf.variable_scope('CLASSIFIER'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            h_layers = [None for _ in range(len(h_layer_sizes))]

            h_layers[0] = tf.layers.dense(x, h_layer_sizes[0], activation=tf.nn.relu)
            for i in range(len(h_layer_sizes)-1):
                h_layers[i+1] = tf.layers.dense(h_layers[i], h_layer_sizes[i+1], activation=tf.nn.relu)
            out = tf.nn.softmax(h_layers[-1])

        return out


def get_label():
    while True:
        for l in range(1,num_actions+1):
            print(actionspace[l-1],' = ', l)
        print('select label: ')
        inp = input()
        if int(inp)<=num_actions and int(inp)>=1:
            return actionspace[int(inp)-1]
        else:
            continue



# HPs
NUM_CHANNELS = 6
num_recordings = 1
batch_size = 32
z_chan = 64
h_size = 64
num_actions = 3
enc_action_space = np.identity(num_actions, dtype=float)
port = '\\.\COM3'


# MODEL
#   PLACEHOLDERs
wm = processer(z_chan, h_size, depth=5, input_size=batch_size)

x1 = tf.placeholder(name='X1', shape=(num_recordings, batch_size, 1), dtype=tf.float32)
x2 = tf.placeholder(name='X2', shape=(num_recordings, batch_size, 1), dtype=tf.float32)
x3 = tf.placeholder(name='X3', shape=(num_recordings, batch_size, 1), dtype=tf.float32)
x4 = tf.placeholder(name='X4', shape=(num_recordings, batch_size, 1), dtype=tf.float32)
x5 = tf.placeholder(name='X5', shape=(num_recordings, batch_size, 1), dtype=tf.float32)
x6 = tf.placeholder(name='X6', shape=(num_recordings, batch_size, 1), dtype=tf.float32)

z = tf.placeholder(name='z', shape=(num_recordings, NUM_CHANNELS, *wm.z_shape), dtype=tf.float32)
label = tf.placeholder(name='label', shape=(num_actions), dtype=tf.float32)

#   ADDITIONAL VARIABLES
gamma = 5
beta = 1

#   FUNCTIONs
mean1, var1 = tf.nn.moments(x1, 0)
mean2, var2 = tf.nn.moments(x2, 0)
mean3, var3 = tf.nn.moments(x3, 0)
mean4, var4 = tf.nn.moments(x4, 0)
mean5, var5 = tf.nn.moments(x5, 0)
mean6, var6 = tf.nn.moments(x6, 0)
nx1 = (x1 - mean1)/tf.sqrt(tf.square(var1)+0.00001) * gamma + beta
nx2 = (x2 - mean2)/tf.sqrt(tf.square(var2)+0.00001) * gamma + beta
nx3 = (x3 - mean3)/tf.sqrt(tf.square(var3)+0.00001) * gamma + beta
nx4 = (x4 - mean4)/tf.sqrt(tf.square(var4)+0.00001) * gamma + beta
nx5 = (x5 - mean5)/tf.sqrt(tf.square(var5)+0.00001) * gamma + beta
nx6 = (x6 - mean6)/tf.sqrt(tf.square(var6)+0.00001) * gamma + beta

xx1, xx2, xx3, xx4, xx5, xx6 = wm.encoder(nx1), wm.encoder(nx2, reuse=True), wm.encoder(nx3, reuse=True), wm.encoder(nx4, reuse=True), wm.encoder(nx5, reuse=True), wm.encoder(nx6, reuse=True)
z = tf.concat(axis=0, values=[xx1,xx2,xx3,xx4,xx5,xx6])
prediction = wm.fully_connected(z, h_layer_sizes=(75, 100, 50, 25, 12 ,num_actions))
prediction_error = sum([(tf.losses.softmax_cross_entropy(label, tf.squeeze(prediction[u]))) for u in range(NUM_CHANNELS)])
optimizer = tf.train.AdamOptimizer(learning_rate=0.003).minimize(prediction_error, var_list=tf.trainable_variables())


# TRAINING
actionspace = ['a','b','n']
training = True
iterations = 100
past_actions = np.empty([0, NUM_CHANNELS])

serial_connection = ser.Serial(port=port, baudrate=9600)
print('serial port: ', serial_connection.name, ' used')

#   LOOP
with tf.Session() as sess:
    if len(actionspace) != num_actions:
        raise(AttributeError)

    sess.run(tf.global_variables_initializer())

    if training:
        #for i in range(iterations):

            # action_label = get_label()
        print(serial_connection.read(6*6))

            # a1, a2, a3, a4, a5, a6 = np.reshape(serial_connection.read(NUM_CHANNELS * batch_size), [NUM_CHANNELS, batch_size])
            # print(a1,a2,a3,a4,a5,a6)
            # _, pred, pred_err = sess.run([optimizer, prediction, prediction_error], feed_dict={'X1:0': a1, 'X2:0': a2, 'X3:0': a3, 'X4:0': a4, 'X5:0': a5, 'X6:0': a6, 'label:0':action_label})
            
            # np.append(past_actions, [a1,a2,a3,a4,a5,a6], 0)
            # # past_actions -> encode in temporal context (RNN) -> "env representation" -> policy network -> action_label_2



