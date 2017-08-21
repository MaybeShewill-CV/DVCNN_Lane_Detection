#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : cnn_util.py
"""
Some dvcnn util functions
"""
import tensorflow as tf
import os
import json
import collections

from DVCNN.model_def import dvcnn_global_variable


def write_dvcnn_model(json_path):
    """
    Write model definition into json file
    :param json_path: path to store model
    :return:
    """
    if _model_json_exist(json_path):
        print('{} already exist'.format(json_path))
        return

    dvcnn_model = _convert_to_ordered_dict(model_dict=dvcnn_global_variable.DVCNN_ARCHITECTURE)
    jsonobj = json.dumps(dvcnn_model, indent=4)
    file = open(json_path, 'w')
    file.write(jsonobj)
    file.close()


def _model_json_exist(json_path):
    """
    Check if the json file exists
    :param json_path:
    :return:
    """
    return os.path.isfile(json_path)


def _convert_to_ordered_dict(model_dict):
    """
    Convert model dict into orderdict
    :param model_dict: model dict
    :return:
    """
    order_dict = collections.OrderedDict()
    for layer_name, layer_para in model_dict.items():
        order_dict[layer_name] = layer_para
    return order_dict


def read_json_model(json_model_path):
    """
    Read json model file into orderddict
    :param json_model_path:
    :return:
    """
    if not os.path.exists(json_model_path):
        raise ValueError('{:s} doesn\'t exist'.format(json_model_path))
    if not json_model_path.endswith('.json'):
        raise ValueError('model file should be a json file')

    with open(json_model_path, 'r') as f:
        jsonobj = json.load(f, object_pairs_hook=collections.OrderedDict)
        f.close()
    return jsonobj


def conv2d(_input, _conv_para, name, reuse=False):
    """
    Define the convolution function
    :param _input:
    :param _conv_para:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        # truncated normal initialize
        init_w = tf.truncated_normal(shape=_conv_para['ksize'], mean=0, stddev=0.02)
        weights = tf.get_variable(name='weights', dtype=tf.float32, initializer=init_w,
                                  trainable=_conv_para['trainable'])
        output = tf.nn.conv2d(_input, weights, _conv_para['strides'], _conv_para['padding'])
        out_channels = _conv_para['ksize'][-1]
        # zero initialize
        init_b = tf.zeros([out_channels])
        bias = tf.get_variable(name='bias', initializer=init_b, dtype=tf.float32, trainable=_conv_para['trainable'])
        output = tf.nn.bias_add(output, bias)
        return output


def activate(_input, _activate_para, name, reuse=False):
    """
    Define the activation function
    :param _input:
    :param _activate_para:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        if _activate_para['method'] == 'RELU':
            return tf.nn.relu(_input, name='Relu_activation')
        elif _activate_para['method'] == 'SIGMOID':
            return tf.nn.sigmoid(_input, name='Sigmoid_activation')
        elif _activate_para['method'] == 'TANH':
            return tf.nn.tanh(_input, name='Tanh_activation')
        else:
            return NotImplementedError


def max_pool(_input, _max_pool_para, name, reuse=False):
    """
    Define the pooling function
    :param _input:
    :param _max_pool_para:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        return tf.nn.max_pool(_input, _max_pool_para['ksize'], _max_pool_para['strides'], _max_pool_para['padding'])


def concat(_input, _concat_para, name):
    """
    Define the concat function
    :param _input:
    :param _concat_para:
    :param name:
    :return:
    """
    return tf.concat(values=_input, axis=_concat_para['axis'], name=name)


def fully_connect(_input, _fc_para, name, reuse=False):
    """
    Define the fully connection function
    :param _input:
    :param _fc_para:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        # truncated normal initialize
        init_w = tf.truncated_normal(shape=_fc_para['ksize'], mean=0, stddev=0.02)
        weights = tf.get_variable(name='weights', initializer=init_w, dtype=tf.float32, trainable=_fc_para['trainable'])
        output = tf.nn.conv2d(_input, weights, _fc_para['strides'], _fc_para['padding'])
        out_channels = _fc_para['ksize'][-1]
        # zero initialize
        init_b = tf.zeros([out_channels])
        bias = tf.get_variable(name='bias', initializer=init_b, dtype=tf.float32, trainable=_fc_para['trainable'])
        output = tf.nn.bias_add(output, bias)
        return output


def batch_norm(_input, name, reuse=False):
    """
    Define the batch normally function
    :param _input:
    :param name:
    :param reuse:
    :return:
    """
    return tf.layers.batch_normalization(_input, name=name, reuse=reuse)
