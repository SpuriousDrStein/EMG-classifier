import tensorflow as tf
import numpy as np
import math as m
import serial as ser
import os
import codecs
from tensorflow.contrib.nn import conv1d_transpose
from tensorflow.contrib.opt import AdamWOptimizer
from tensorflow.contrib import autograph

class processer:
    
    def __init__(self, z_channels, depth=5, input_size=30, batch_size=1): # input_size x 1
        self.input_size = input_size
        self.depth = depth
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
            nn[0] = tf.layers.average_pooling1d(nn[0], pool_size, 2)
            for d in range(1, self.depth-1):
                nn[d] = tf.layers.conv1d(nn[d-1], c[d+1], k_size, padding='SAME', kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.constant(0), activation=tf.nn.relu)
                nn[d] = tf.layers.average_pooling1d(nn[d], pool_size, 2)
            nn[-1] = tf.layers.conv1d(nn[-2], c[-1], k_size, padding='SAME', kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.constant(0), activation=tf.nn.relu)
            nn[-1] = tf.layers.average_pooling1d(nn[-1], pool_size, 2)
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


def get_label(actionspace, action_one_hot, exit_value=None):
    while True:
        for l in range(1,num_actions+1):
            print(actionspace[l-1],' = ', l)
        print('select label: ')
        print('or exit by: ', exit_value)
        inp = input()
        if inp == exit_value:
            exit(0)
        if int(inp)<=num_actions and int(inp)>=1:
            return action_one_hot[int(inp)-1]
        else:
            continue



# # HPs
# NUM_CHANNELS = 6
# num_recordings = 1
# batch_size = 256 * 2
# z_chan = 64
# h_size = 64
# num_actions = 3
# port = '\\.\COM3'

# # TRAINING
# actionspace = ['a','b','n']
# action_one_hot = np.identity(len(actionspace))
# training = True
# load = False
# iterations = 100
# past_actions = np.empty([1, NUM_CHANNELS, num_recordings, batch_size, 1])
# saver = tf.train.Saver()
# pred_err = 9999
# local_save_path = '/saved_models/model_save.ckpt'



# # MODEL
# #   PLACEHOLDERs
# wm = processer(z_chan, h_size, depth=5, input_size=batch_size)

# x1 = tf.placeholder(name='X1', shape=(num_recordings, batch_size, 1), dtype=tf.float32)
# x2 = tf.placeholder(name='X2', shape=(num_recordings, batch_size, 1), dtype=tf.float32)
# x3 = tf.placeholder(name='X3', shape=(num_recordings, batch_size, 1), dtype=tf.float32)
# x4 = tf.placeholder(name='X4', shape=(num_recordings, batch_size, 1), dtype=tf.float32)
# x5 = tf.placeholder(name='X5', shape=(num_recordings, batch_size, 1), dtype=tf.float32)
# x6 = tf.placeholder(name='X6', shape=(num_recordings, batch_size, 1), dtype=tf.float32)

# z = tf.placeholder(name='z', shape=(num_recordings, NUM_CHANNELS, *wm.z_shape), dtype=tf.float32)
# label = tf.placeholder(name='label', shape=(num_actions), dtype=tf.float32)

# #   ADDITIONAL VARIABLES
# gamma = 5
# beta = 1

# #   FUNCTIONs
# mean1, var1 = tf.nn.moments(x1, 0)
# mean2, var2 = tf.nn.moments(x2, 0)
# mean3, var3 = tf.nn.moments(x3, 0)
# mean4, var4 = tf.nn.moments(x4, 0)
# mean5, var5 = tf.nn.moments(x5, 0)
# mean6, var6 = tf.nn.moments(x6, 0)
# nx1 = (x1 - mean1)/tf.sqrt(tf.square(var1)+0.00001) * gamma + beta
# nx2 = (x2 - mean2)/tf.sqrt(tf.square(var2)+0.00001) * gamma + beta
# nx3 = (x3 - mean3)/tf.sqrt(tf.square(var3)+0.00001) * gamma + beta
# nx4 = (x4 - mean4)/tf.sqrt(tf.square(var4)+0.00001) * gamma + beta
# nx5 = (x5 - mean5)/tf.sqrt(tf.square(var5)+0.00001) * gamma + beta
# nx6 = (x6 - mean6)/tf.sqrt(tf.square(var6)+0.00001) * gamma + beta

# xx1, xx2, xx3, xx4, xx5, xx6 = wm.encoder(nx1), wm.encoder(nx2, reuse=True), wm.encoder(nx3, reuse=True), wm.encoder(nx4, reuse=True), wm.encoder(nx5, reuse=True), wm.encoder(nx6, reuse=True)
# z = tf.layers.flatten(tf.concat(axis=0, values=[xx1,xx2,xx3,xx4,xx5,xx6]))
# print('flattend z shape: ', z.shape)
# prediction = wm.fully_connected(z, h_layer_sizes=(h_size*NUM_CHANNELS, int((h_size*NUM_CHANNELS)/2), int((h_size*NUM_CHANNELS)/4), int((h_size*NUM_CHANNELS)/6), int((h_size*NUM_CHANNELS)/8) ,num_actions))
# print('prediction shape: ', prediction.shape)
# prediction_error = tf.reduce_sum([(tf.losses.softmax_cross_entropy(label, tf.squeeze(prediction[u]))) for u in range(NUM_CHANNELS)])
# optimizer = AdamWOptimizer(learning_rate=0.001, weight_decay=0.3).minimize(prediction_error, var_list=tf.trainable_variables())


# serial_connection = ser.Serial(port=port, baudrate=9600)
# print('serial port: ', serial_connection.name, ' used. \nfirst hex bit: ', serial_connection.read(1))

# #   LOOP
# with tf.Session() as sess:
#     if len(actionspace) != num_actions:
#         raise(AttributeError)

#     if load:
#         print('load model')
#         saver.restore(sess, local_save_path)

#     sess.run(tf.global_variables_initializer())

#     if training:
#         for i in range(iterations):

#             print('current itteration:  ', i)
#             print('current error:       ', pred_err)
#             action_label = get_label(actionspace, action_one_hot, exit_value='0')

#             c = serial_connection.read(NUM_CHANNELS*batch_size*4)
#             c = codecs.decode(c, 'ascii')
#             c = c.split('\n')
#             c = [d.split('\t') for d in c[0:-1]]

#             for i in range(0,len(c)):
#                 for j in range(0, len(c[i])):
#                     c[i][j] = float.fromhex(c[i][j])

#             c = np.array(c, dtype=float)
#             c1, c2, c3, c4, c5, c6 = [np.expand_dims(np.expand_dims(c[:,i], 0), -1) for i in range(NUM_CHANNELS)]
#             _, pred, pred_err = sess.run([optimizer, prediction, prediction_error], feed_dict={'X1:0': c1, 'X2:0': c2, 'X3:0': c3, 'X4:0': c4, 'X5:0': c5, 'X6:0': c6, 'label:0':action_label})

#             sp = saver.save(sess, local_save_path)
#             print('saved to path: ', sp)

#             # np.append(past_actions, np.expand_dims([c1, c2, c3, c4, c5, c6], 0), 0)
#             # past_actions -> encode in temporal context (RNN) -> "env representation" -> policy network -> action_label_2


