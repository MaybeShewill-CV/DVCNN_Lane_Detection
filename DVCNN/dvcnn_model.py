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
        self.__conv2d = cnn_util.conv2d
        self.__max_pool = cnn_util.max_pool
        self.__activate = cnn_util.activate
        self.__fully_connect = cnn_util.fully_connect
        self.__concat = cnn_util.concat
        self.__batch_norm = cnn_util.batch_norm
        self.__dvcnn_architecture = cnn_util.read_json_model(json_model_path=json_model_path)

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

    def build_dvcnn_test(self, top_view_input, front_view_input):
        """
        Build dvcnn model
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
