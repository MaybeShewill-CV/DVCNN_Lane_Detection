#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : dvcnn_global_variable.py
"""
Some dvcnn global parameters, you should not change it easily
"""
from Global_Configuration.config import cfg


DVCNN_ARCHITECTURE = {
    'conv1': {
        'ksize': [5, 5, 3, 16],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'relu1': {
        'method': 'RELU'
        },
    'pool1': {
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'conv2_front': {
        'ksize': [5, 5, 16, 16],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'conv2_top': {
        'ksize': [5, 5, 3, 16],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'relu2': {
        'method': 'RELU'
    },
    'pool2': {
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'conv3': {
        'ksize': [5, 5, 16, 32],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'relu3': {
        'method': 'RELU'
    },
    'pool3': {
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'conv4': {
        'ksize': [5, 5, 32, 32],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'relu4': {
        'method': 'RELU'
    },
    'pool4': {
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'conv5': {
        'ksize': [5, 5, 32, 64],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'relu5': {
        'method': 'RELU'
    },
    'pool5': {
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'fc6': {
        'ksize': [4, 4, 64, 256],
        'strides': [1, 4, 4, 1],
        'padding': 'VALID',
        'trainable': True
    },
    'relu6': {
        'method': 'RELU'
    },
    'concat7': {
        'axis': 3
    },
    'fc8': {
        'ksize': [1, 1, 512, 2],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'trainable': True
    }
}


DVCNN_AUGMENTATION_DICTS = {
    'whiten': {
        'mean_value': [103.939, 116.779, 123.68]
    },
    'random_brightness': {
        'brightness': cfg.TRAIN.RANDOM_BRIGHTNESS_VALUE
    },
    'random_contrast': {
        'lower_factor': cfg.TRAIN.RANDOM_CONTRAST_LOWER_VALUE,
        'upper_factor': cfg.TRAIN.RANDOM_CONTRAST_HIGHER_VALUE
    },
    'random_crop': {
        'crop_size': cfg.TRAIN.RANDOM_CROP_VALUE
    }
}
