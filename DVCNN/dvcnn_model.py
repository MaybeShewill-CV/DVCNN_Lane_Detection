#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : dvcnn_model.py
"""
Construct the DVCNN model
"""
import tensorflow as tf
from DVCNN import cnn_util


class DVCNNBuilder(object):
    def __init__(self, json_model_path):
        self.__dvcnn_architecture = cnn_util.read_json_model(json_model_path=json_model_path)
        return

    @staticmethod
    def __conv2d(_input, _conv_para, name, reuse=False):
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

    @staticmethod
    def __activate(_input, _activate_para, name, reuse=False):
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

    @staticmethod
    def __max_pool(_input, _max_pool_para, name, reuse=False):
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

    @staticmethod
    def __concat(_input, _concat_para, name):
        """
        Define the concat function
        :param _input:
        :param _concat_para:
        :param name:
        :return:
        """
        return tf.concat(values=_input, axis=_concat_para['axis'], name=name)

    @staticmethod
    def __fully_connect(_input, _fc_para, name, reuse=False):
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
            weights = tf.get_variable(name='weights', initializer=init_w, dtype=tf.float32,
                                      trainable=_fc_para['trainable'])
            output = tf.nn.conv2d(_input, weights, _fc_para['strides'], _fc_para['padding'])
            out_channels = _fc_para['ksize'][-1]
            # zero initialize
            init_b = tf.zeros([out_channels])
            bias = tf.get_variable(name='bias', initializer=init_b, dtype=tf.float32, trainable=_fc_para['trainable'])
            output = tf.nn.bias_add(output, bias)
            return output

    @staticmethod
    def __batch_norm(_input, name, reuse=False):
        """
        Define the batch normally function
        :param _input:
        :param name:
        :param reuse:
        :return:
        """
        return tf.layers.batch_normalization(_input, name=name, reuse=reuse)

    def build_dvcnn(self, top_view_input, front_view_input):
        """
        Build dvcnn model
        :param top_view_input: top view input tensor normalized into 64*64
        :param front_view_input: front view input tensor normalized into 128*128
        :return:softmax logits with 2cls [not_road_line, is_road_line]
        """
        # front view input begins at conv1 and top view input begins at conv2
        # Stage 1
        front_conv1 = self.__conv2d(_input=front_view_input, _conv_para=self.__dvcnn_architecture['conv1'],
                                    name='conv1', reuse=False)
        front_bn1 = self.__batch_norm(_input=front_conv1, name='bn1', reuse=False)
        front_relu1 = self.__activate(_input=front_bn1, _activate_para=self.__dvcnn_architecture['relu1'],
                                      name='relu1', reuse=False)
        front_pool1 = self.__max_pool(_input=front_relu1, _max_pool_para=self.__dvcnn_architecture['pool1'],
                                      name='pool1', reuse=False)

        # Stage 2
        front_conv2 = self.__conv2d(_input=front_pool1, _conv_para=self.__dvcnn_architecture['conv2_front'],
                                    name='conv2_front', reuse=False)
        front_bn2 = self.__batch_norm(_input=front_conv2, name='bn2_front', reuse=False)
        front_relu2 = self.__activate(_input=front_bn2, _activate_para=self.__dvcnn_architecture['relu2'],
                                      name='relu2', reuse=False)
        front_pool2 = self.__max_pool(_input=front_relu2, _max_pool_para=self.__dvcnn_architecture['pool2'],
                                      name='pool2', reuse=False)

        top_conv2 = self.__conv2d(_input=top_view_input, _conv_para=self.__dvcnn_architecture['conv2_top'],
                                  name='conv2_top', reuse=False)
        top_bn2 = self.__batch_norm(_input=top_conv2, name='bn2_top', reuse=False)
        top_relu2 = self.__activate(_input=top_bn2, _activate_para=self.__dvcnn_architecture['relu2'],
                                    name='relu2', reuse=True)
        top_pool2 = self.__max_pool(_input=top_relu2, _max_pool_para=self.__dvcnn_architecture['pool2'],
                                    name='pool2', reuse=True)

        # Stage 3
        front_conv3 = self.__conv2d(_input=front_pool2, _conv_para=self.__dvcnn_architecture['conv3'],
                                    name='conv3', reuse=False)
        front_bn3 = self.__batch_norm(_input=front_conv3, name='bn3', reuse=False)
        front_relu3 = self.__activate(_input=front_bn3, _activate_para=self.__dvcnn_architecture['relu3'],
                                      name='relu3', reuse=False)
        front_pool3 = self.__max_pool(_input=front_relu3, _max_pool_para=self.__dvcnn_architecture['pool3'],
                                      name='pool3', reuse=False)

        top_conv3 = self.__conv2d(_input=top_pool2, _conv_para=self.__dvcnn_architecture['conv3'],
                                  name='conv3', reuse=True)
        top_bn3 = self.__batch_norm(_input=top_conv3, name='bn3', reuse=True)
        top_relu3 = self.__activate(_input=top_bn3, _activate_para=self.__dvcnn_architecture['relu3'],
                                    name='relu3', reuse=True)
        top_pool3 = self.__max_pool(_input=top_relu3, _max_pool_para=self.__dvcnn_architecture['pool3'],
                                    name='pool3', reuse=True)

        # Stage 4
        front_conv4 = self.__conv2d(_input=front_pool3, _conv_para=self.__dvcnn_architecture['conv4'],
                                    name='conv4', reuse=False)
        front_bn4 = self.__batch_norm(_input=front_conv4, name='bn4', reuse=False)
        front_relu4 = self.__activate(_input=front_bn4, _activate_para=self.__dvcnn_architecture['relu4'],
                                      name='relu4', reuse=False)
        front_pool4 = self.__max_pool(_input=front_relu4, _max_pool_para=self.__dvcnn_architecture['pool4'],
                                      name='pool4', reuse=False)

        top_conv4 = self.__conv2d(_input=top_pool3, _conv_para=self.__dvcnn_architecture['conv4'],
                                  name='conv4', reuse=True)
        top_bn4 = self.__batch_norm(_input=top_conv4, name='bn4', reuse=True)
        top_relu4 = self.__activate(_input=top_bn4, _activate_para=self.__dvcnn_architecture['relu4'],
                                    name='relu4', reuse=True)
        top_pool4 = self.__max_pool(_input=top_relu4, _max_pool_para=self.__dvcnn_architecture['pool4'],
                                    name='pool4', reuse=True)

        # Stage 5
        front_conv5 = self.__conv2d(_input=front_pool4, _conv_para=self.__dvcnn_architecture['conv5'],
                                    name='conv5', reuse=False)
        front_bn5 = self.__batch_norm(_input=front_conv5, name='bn5', reuse=False)
        front_relu5 = self.__activate(_input=front_bn5, _activate_para=self.__dvcnn_architecture['relu5'],
                                      name='relu5', reuse=False)
        front_pool5 = self.__max_pool(_input=front_relu5, _max_pool_para=self.__dvcnn_architecture['pool5'],
                                      name='pool5', reuse=False)

        top_conv5 = self.__conv2d(_input=top_pool4, _conv_para=self.__dvcnn_architecture['conv5'],
                                  name='conv5', reuse=True)
        top_bn5 = self.__batch_norm(_input=top_conv5, name='bn5', reuse=True)
        top_relu5 = self.__activate(_input=top_bn5, _activate_para=self.__dvcnn_architecture['relu5'],
                                    name='relu5', reuse=True)
        top_pool5 = self.__max_pool(_input=top_relu5, _max_pool_para=self.__dvcnn_architecture['pool5'],
                                    name='pool5', reuse=True)

        # Stage 6
        front_fc6 = self.__fully_connect(_input=front_pool5, _fc_para=self.__dvcnn_architecture['fc6'],
                                         name='fc6', reuse=False)
        front_bn6 = self.__batch_norm(_input=front_fc6, name='bn6', reuse=False)
        front_relu6 = self.__activate(_input=front_bn6, _activate_para=self.__dvcnn_architecture['relu6'],
                                      name='relu6', reuse=False)

        top_fc6 = self.__fully_connect(_input=top_pool5, _fc_para=self.__dvcnn_architecture['fc6'],
                                       name='fc6', reuse=True)
        top_bn6 = self.__batch_norm(_input=top_fc6, name='bn6', reuse=True)
        top_relu6 = self.__activate(_input=top_bn6, _activate_para=self.__dvcnn_architecture['relu6'],
                                    name='relu6', reuse=True)

        # Stage 7
        concat7 = self.__concat(_input=[front_relu6, top_relu6], _concat_para=self.__dvcnn_architecture['concat7'],
                                name='concat7')

        # Stage 8
        fc8 = self.__fully_connect(_input=concat7, _fc_para=self.__dvcnn_architecture['fc8'],
                                   name='fc8', reuse=False)

        # Convert fc8 from matrix into a vector
        out_put = tf.reshape(tensor=fc8, shape=[-1, self.__dvcnn_architecture['fc8']['ksize'][-1]])

        return out_put

    def build_dvcnn_val(self, top_view_input, front_view_input):
        """
        Build dvcnn model for evaluation
        :param top_view_input: top view input tensor normalized into 64*64
        :param front_view_input: front view input tensor normalized into 128*128
        :return:softmax logits with 2cls [not_road_line, is_road_line]
        """
        # front view input begins at conv1 and top view input begins at conv2
        # Stage 1
        front_conv1 = self.__conv2d(_input=front_view_input, _conv_para=self.__dvcnn_architecture['conv1'],
                                    name='conv1', reuse=True)
        front_bn1 = self.__batch_norm(_input=front_conv1, name='bn1', reuse=True)
        front_relu1 = self.__activate(_input=front_bn1, _activate_para=self.__dvcnn_architecture['relu1'],
                                      name='relu1', reuse=True)
        front_pool1 = self.__max_pool(_input=front_relu1, _max_pool_para=self.__dvcnn_architecture['pool1'],
                                      name='pool1', reuse=True)

        # Stage 2
        front_conv2 = self.__conv2d(_input=front_pool1, _conv_para=self.__dvcnn_architecture['conv2_front'],
                                    name='conv2_front', reuse=True)
        front_bn2 = self.__batch_norm(_input=front_conv2, name='bn2_front', reuse=True)
        front_relu2 = self.__activate(_input=front_bn2, _activate_para=self.__dvcnn_architecture['relu2'],
                                      name='relu2', reuse=True)
        front_pool2 = self.__max_pool(_input=front_relu2, _max_pool_para=self.__dvcnn_architecture['pool2'],
                                      name='pool2', reuse=True)

        top_conv2 = self.__conv2d(_input=top_view_input, _conv_para=self.__dvcnn_architecture['conv2_top'],
                                  name='conv2_top', reuse=True)
        top_bn2 = self.__batch_norm(_input=top_conv2, name='bn2_top', reuse=True)
        top_relu2 = self.__activate(_input=top_bn2, _activate_para=self.__dvcnn_architecture['relu2'],
                                    name='relu2', reuse=True)
        top_pool2 = self.__max_pool(_input=top_relu2, _max_pool_para=self.__dvcnn_architecture['pool2'],
                                    name='pool2', reuse=True)

        # Stage 3
        front_conv3 = self.__conv2d(_input=front_pool2, _conv_para=self.__dvcnn_architecture['conv3'],
                                    name='conv3', reuse=True)
        front_bn3 = self.__batch_norm(_input=front_conv3, name='bn3', reuse=True)
        front_relu3 = self.__activate(_input=front_bn3, _activate_para=self.__dvcnn_architecture['relu3'],
                                      name='relu3', reuse=True)
        front_pool3 = self.__max_pool(_input=front_relu3, _max_pool_para=self.__dvcnn_architecture['pool3'],
                                      name='pool3', reuse=True)

        top_conv3 = self.__conv2d(_input=top_pool2, _conv_para=self.__dvcnn_architecture['conv3'],
                                  name='conv3', reuse=True)
        top_bn3 = self.__batch_norm(_input=top_conv3, name='bn3', reuse=True)
        top_relu3 = self.__activate(_input=top_bn3, _activate_para=self.__dvcnn_architecture['relu3'],
                                    name='relu3', reuse=True)
        top_pool3 = self.__max_pool(_input=top_relu3, _max_pool_para=self.__dvcnn_architecture['pool3'],
                                    name='pool3', reuse=True)

        # Stage 4
        front_conv4 = self.__conv2d(_input=front_pool3, _conv_para=self.__dvcnn_architecture['conv4'],
                                    name='conv4', reuse=True)
        front_bn4 = self.__batch_norm(_input=front_conv4, name='bn4', reuse=True)
        front_relu4 = self.__activate(_input=front_bn4, _activate_para=self.__dvcnn_architecture['relu4'],
                                      name='relu4', reuse=True)
        front_pool4 = self.__max_pool(_input=front_relu4, _max_pool_para=self.__dvcnn_architecture['pool4'],
                                      name='pool4', reuse=True)

        top_conv4 = self.__conv2d(_input=top_pool3, _conv_para=self.__dvcnn_architecture['conv4'],
                                  name='conv4', reuse=True)
        top_bn4 = self.__batch_norm(_input=top_conv4, name='bn4', reuse=True)
        top_relu4 = self.__activate(_input=top_bn4, _activate_para=self.__dvcnn_architecture['relu4'],
                                    name='relu4', reuse=True)
        top_pool4 = self.__max_pool(_input=top_relu4, _max_pool_para=self.__dvcnn_architecture['pool4'],
                                    name='pool4', reuse=True)

        # Stage 5
        front_conv5 = self.__conv2d(_input=front_pool4, _conv_para=self.__dvcnn_architecture['conv5'],
                                    name='conv5', reuse=True)
        front_bn5 = self.__batch_norm(_input=front_conv5, name='bn5', reuse=True)
        front_relu5 = self.__activate(_input=front_bn5, _activate_para=self.__dvcnn_architecture['relu5'],
                                      name='relu5', reuse=True)
        front_pool5 = self.__max_pool(_input=front_relu5, _max_pool_para=self.__dvcnn_architecture['pool5'],
                                      name='pool5', reuse=True)

        top_conv5 = self.__conv2d(_input=top_pool4, _conv_para=self.__dvcnn_architecture['conv5'],
                                  name='conv5', reuse=True)
        top_bn5 = self.__batch_norm(_input=top_conv5, name='bn5', reuse=True)
        top_relu5 = self.__activate(_input=top_bn5, _activate_para=self.__dvcnn_architecture['relu5'],
                                    name='relu5', reuse=True)
        top_pool5 = self.__max_pool(_input=top_relu5, _max_pool_para=self.__dvcnn_architecture['pool5'],
                                    name='pool5', reuse=True)

        # Stage 6
        front_fc6 = self.__fully_connect(_input=front_pool5, _fc_para=self.__dvcnn_architecture['fc6'],
                                         name='fc6', reuse=True)
        front_bn6 = self.__batch_norm(_input=front_fc6, name='bn6', reuse=True)
        front_relu6 = self.__activate(_input=front_bn6, _activate_para=self.__dvcnn_architecture['relu6'],
                                      name='relu6', reuse=True)

        top_fc6 = self.__fully_connect(_input=top_pool5, _fc_para=self.__dvcnn_architecture['fc6'],
                                       name='fc6', reuse=True)
        top_bn6 = self.__batch_norm(_input=top_fc6, name='bn6', reuse=True)
        top_relu6 = self.__activate(_input=top_bn6, _activate_para=self.__dvcnn_architecture['relu6'],
                                    name='relu6', reuse=True)

        # Stage 7
        concat7 = self.__concat(_input=[front_relu6, top_relu6], _concat_para=self.__dvcnn_architecture['concat7'],
                                name='concat7')

        # Stage 8
        fc8 = self.__fully_connect(_input=concat7, _fc_para=self.__dvcnn_architecture['fc8'],
                                   name='fc8', reuse=True)

        # Convert fc8 from matrix into a vector
        out_put = tf.reshape(tensor=fc8, shape=[-1, self.__dvcnn_architecture['fc8']['ksize'][-1]])

        return out_put
