# -*- coding: utf-8 -*-
""" Very Deep De-Convolutional Networks for Large-Scale Visual Recognition / Segmentation.

Applying VGG 16-layers convolutional +  reverse VGG-16 Layers

"""
    

#from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, conv_2d_transpose
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tensorflow as tf

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist


num_classes = 17


def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)],i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out




# Shape = List of dimensions = [None, 224, 224, 3]
def BUILD_DCONVNET(shape):
    # Building deconvolutional network
    # Building 'VGG Network'
    network = input_data(shape)

    conv_1_1 = conv_2d(network, 64, 3, activation='relu')
    conv_1_2 = conv_2d(conv_1_1, 64, 3, activation='relu')
    
    pool_1, pool_1_argmax = tf.nn.max_pool_with_argmax(conv_1_2,
                                                       ksize=[1, 2, 2, 1],
                                                       strides=[1, 2, 2, 1],
                                                       padding='SAME')
    pool_1 = max_pool_2d(conv_1_2, 2, strides=2)
    
################################################################################

    conv_2_1 = conv_2d(pool_1, 128, 3, activation='relu')
    conv_2_2 = conv_2d(conv_2_1, 128, 3, activation='relu')
    pool_2, pool_2_argmax = tf.nn.max_pool_with_argmax(conv_2_2,
                                                       ksize=[1, 2, 2, 1],
                                                       strides=[1, 2, 2, 1],
                                                       padding='SAME')

    pool_2 = max_pool_2d(conv_2_2, 2, strides=2)

################################################################################

    conv_3_1 = conv_2d(pool_2, 256, 3, activation='relu')
    conv_3_2 = conv_2d(conv_3_1, 256, 3, activation='relu')
    conv_3_3 = conv_2d(conv_3_2, 256, 3, activation='relu')
    pool_3, pool_3_argmax = tf.nn.max_pool_with_argmax(conv_3_3,
                                                       ksize=[1, 2, 2, 1],
                                                       strides=[1, 2, 2, 1],
                                                       padding='SAME')

    pool_3 = max_pool_2d(conv_3_3, 2, strides=2)

################################################################################

    conv_4_1 = conv_2d(pool_3, 512, 3, activation='relu')
    conv_4_2 = conv_2d(conv_4_1, 512, 3, activation='relu')
    conv_4_3 = conv_2d(conv_4_2, 512, 3, activation='relu')
    pool_4, pool_4_argmax = tf.nn.max_pool_with_argmax(conv_4_3,
                                                       ksize=[1, 2, 2, 1],
                                                       strides=[1, 2, 2, 1],
                                                       padding='SAME')

    pool_4 = max_pool_2d(conv_4_3, 2, strides=2)

################################################################################

    conv_5_1 = conv_2d(pool_4, 512, 3, activation='relu')
    conv_5_2 = conv_2d(conv_5_1, 512, 3, activation='relu')
    conv_5_3 = conv_2d(conv_5_2, 512, 3, activation='relu')
    pool_5, pool_5_argmax = tf.nn.max_pool_with_argmax(conv_5_3,
                                                       ksize=[1, 2, 2, 1],
                                                       strides=[1, 2, 2, 1],
                                                       padding='SAME')

    pool_5 = max_pool_2d(conv_5_3, 2, strides=2)

################################################################################

    fc_6 = conv_2d(pool_5, 4096, 7, activation='relu')

################################################################################
    
    fc_7 = conv_2d(fc_6, 4096, 1, activation='relu')

################################################################################

    deconv_fc_6 = conv_2d_transpose(fc_7,512,7,[7,7,512], activation='relu')

################################################################################

    unpool_5 = unpool(deconv_fc_6)
    deconv_5_3 = conv_2d_transpose(unpool_5,512,3,[14,14,512], activation='relu')
    deconv_5_2 = conv_2d_transpose(deconv_5_3,512,3,[14,14,512], activation='relu')
    deconv_5_1 = conv_2d_transpose(deconv_5_2,512,3,[14,14,512], activation='relu')

################################################################################

    unpool_4 = unpool(deconv_5_1)
    deconv_4_3 = conv_2d_transpose(unpool_4,512,3,[28,28,512], activation='relu')
    deconv_4_2 = conv_2d_transpose(deconv_4_3,512,3,[28,28,512], activation='relu')
    deconv_4_1 = conv_2d_transpose(deconv_4_2,256,3,[28,28,256], activation='relu')

################################################################################

    unpool_3 = unpool(deconv_4_1)
    deconv_3_3 = conv_2d_transpose(unpool_3,256,3,[56,56,256], activation='relu')
    deconv_3_2 = conv_2d_transpose(deconv_3_3,256,3,[56,56,256], activation='relu')
    deconv_3_1 = conv_2d_transpose(deconv_3_2,128,3,[56,56,128], activation='relu')

################################################################################

    unpool_2 = unpool(deconv_3_1)
    deconv_2_2 = conv_2d_transpose(unpool_2,128,3,[112,112,128], activation='relu')
    deconv_2_1 = conv_2d_transpose(deconv_2_2,64,3,[112,112,64], activation='relu')

################################################################################

    unpool_1 = unpool(deconv_2_1)
    deconv_1_2 = conv_2d_transpose(unpool_2,64,3,[224,224,64], activation='relu')
    deconv_1_1 = conv_2d_transpose(deconv_1_2,64,3,[224,224,64], activation='relu')

    network = fully_connected(deconv_1_1, num_classes, activation='softmax', scope='fc8',
                                restore=False)

    return network


model_path = "vgg16.tflearn"
#data_dir = "/path/to/your/data"
shape=[None, 224, 224, 3]
softmax = BUILD_DCONVNET(shape)

momentum = Momentum(learning_rate=0.01, momentum=0.9, lr_decay=0.0005)
regression = regression(softmax, optimizer=momentum,
                     loss='categorical_crossentropy',
                     learning_rate=0.001,restore=False)


model = tflearn.DNN(regression, checkpoint_path='vgg-finetuning',
                    max_checkpoints=3, tensorboard_verbose=2,
                    tensorboard_dir="./logs")

model_file = os.path.join(model_path, "vgg16.tflearn")
model.load(model_file, weights_only=True)

# Start finetuning
model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_epoch=False,
          snapshot_step=200, run_id='vgg-finetuning')

model.save('VGG16-finetune-bird.tfl')
print("Network trained and saved as bird-classifier.tfl!")
