# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn
import pdb
# Inception-Renset-A
  
def inference(images, mask_res4, mask, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):

    batch_norm_params = {
        # Decay for the moving averages.
        'is_training': phase_train,
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return my_simple(images, mask_res4, mask, is_training=phase_train, bottleneck_layer_size=bottleneck_layer_size)


def my_simple(images, mask_res4, mask, is_training,  dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='SingleNet'):
    end_points = {}

    def block_ds(inputs, filters):
        conv1 = slim.conv2d(inputs, filters, 3, activation_fn=tf.nn.relu, padding = 'valid')
        pool1 = slim.max_pool2d(conv1, 2, stride=2)
        conv2a = slim.conv2d(pool1, filters, 3, activation_fn=tf.nn.relu,padding = 'same')
        conv2b = slim.conv2d(conv2a, filters, 3, activation_fn=tf.nn.relu, padding = 'same')
        res2 = tf.add(conv2b, pool1)
        return res2

    conv1 = slim.conv2d(images, 32, 3, activation_fn=tf.nn.relu,  padding = 'valid')

    res2 = block_ds(conv1, 64)
    res3 = block_ds(res2, 128)
    res4 = block_ds(res3, 128)
 
    end_points['res4'] = res4   
    shape = res4.get_shape().as_list()
    height = shape[1]
    width = shape[2]
    channel = int(shape[3])
    mask_res4 = tf.tile(mask_res4, [1,height, width, 1])
 
    res4 = tf.multiply(res4,mask_res4)


    res5 = block_ds(res4, 128)
    end_points['res5'] = res5
    shape = res5.get_shape().as_list()
    height = shape[1]
    width = shape[2]
    channel = int(shape[3])
    mask = tf.tile(mask, [1,height, width, 1])
    res5 = tf.multiply(res5,mask)

    res5_flatten = slim.flatten(res5)
    fc7 =  slim.fully_connected(res5_flatten, bottleneck_layer_size)
    end_points['fc7'] = fc7

    return fc7, end_points
  
