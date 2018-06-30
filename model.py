from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class Discriminator(object):

    def __init__(self, name):
        self.name = name

    def create_main_graph(self, image):
        with tf.variable_scope(self.name):
            with tf.variable_scope('main_scope', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('layer1'):
                    conv = tf.layers.conv2d(
                        inputs=image,
                        filters=16,
                        kernel_size=[5, 5],
                        padding='same',
                        activation=None,
                        name='conv')
                    norm = tf.layers.batch_normalization(conv, name='norm')
                    active = tf.nn.leaky_relu(norm, name='active')
                    pool = tf.layers.average_pooling2d(
                        inputs=active, pool_size=[2, 2], strides=2, name='pool')
                with tf.variable_scope('layer2'):
                    conv = tf.layers.conv2d(
                        inputs=pool,
                        filters=16,
                        kernel_size=[5, 5],
                        padding='same',
                        activation=None,
                        name='conv')
                    norm = tf.layers.batch_normalization(conv, name='norm')
                    active = tf.nn.leaky_relu(norm, name='active')
                    pool = tf.layers.average_pooling2d(
                        inputs=active, pool_size=[2, 2], strides=2, name='pool')
                with tf.variable_scope('layer3'):
                    conv = tf.layers.conv2d(
                        inputs=pool,
                        filters=16,
                        kernel_size=[5, 5],
                        padding='same',
                        activation=None,
                        name='conv')
                    norm = tf.layers.batch_normalization(conv, name='norm')
                    active = tf.nn.leaky_relu(norm, name='active')
                    pool = tf.layers.average_pooling2d(
                        inputs=active, pool_size=[2, 2], strides=2, name='pool')
                with tf.variable_scope('fully_connected'):
                    shape = pool.shape
                    reshape = tf.reshape(pool,
                                         [-1, shape[1] * shape[2] * shape[3]])
                    fully_connected = tf.layers.dense(
                        inputs=reshape, units=128, name='dense')
                    norm = tf.layers.batch_normalization(
                        fully_connected, name='norm')
                    active = tf.nn.leaky_relu(norm)
                with tf.variable_scope('output'):
                    logits = tf.layers.dense(
                        inputs=active, units=1, name='logits')

        return logits

    def get_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='{}/main_scope'.format(self.name))


class Generator(object):

    def __init__(self, name):
        self.name = name

    def create_main_graph(self, z):
        with tf.variable_scope(self.name):
            with tf.variable_scope('main_scope', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('fully_connected'):
                    fc = tf.layers.dense(
                        inputs=z, units=16 * 16 * 2, name='dense')
                    norm = tf.layers.batch_normalization(fc, name='norm')
                    active = tf.nn.leaky_relu(norm)
                with tf.variable_scope('layer1'):
                    # 32 * 32 * 16
                    reshape = tf.reshape(active, [-1, 16, 16, 2])
                    deconv = tf.layers.conv2d_transpose(
                        inputs=reshape,
                        filters=16,
                        kernel_size=(5, 5),
                        strides=(2, 2),
                        activation=None,
                        padding='same',
                        name='deconv')
                    norm = tf.layers.batch_normalization(deconv, name='norm')
                    active = tf.nn.leaky_relu(norm, name='active')
                with tf.variable_scope('layer2'):
                    # 64 * 64 * 16
                    deconv = tf.layers.conv2d_transpose(
                        inputs=active,
                        filters=16,
                        kernel_size=(5, 5),
                        strides=(2, 2),
                        activation=None,
                        padding='same',
                        name='deconv')
                    norm = tf.layers.batch_normalization(deconv, name='norm')
                    active = tf.nn.leaky_relu(norm, name='active')
                with tf.variable_scope('output'):
                    # 128 * 128 * 4
                    deconv = tf.layers.conv2d_transpose(
                        inputs=active,
                        filters=4,
                        kernel_size=(5, 5),
                        strides=(2, 2),
                        activation=None,
                        padding='same',
                        name='deconv')
                    output = tf.sigmoid(deconv)
        return output

    def get_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='{}/main_scope'.format(self.name))
