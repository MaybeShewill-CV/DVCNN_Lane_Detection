#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : filter_util.py
"""
Implementation of weight hat-like filter
"""
import tensorflow as tf
import numpy as np


class WHatLikeFilter(object):
    def __init__(self, filter_size):
        self.__ksize = filter_size  # [height, width]
        return

    def __str__(self):
        return 'WHatLikeFilter with filter size [{:d}, {:d}]'.format(self.__ksize[0], self.__ksize[1])

    @staticmethod
    def __init_filter_weights(filter_size):
        """
        Use ndarray to initialize the weights
        :return:
        """
        filter_width = filter_size[1]
        filter_height = filter_size[0]
        weights_left = np.ones(shape=[filter_height, filter_width, 1, 1])
        weights_middle = np.ones(shape=[filter_height, filter_width, 1, 1])
        weights_right = np.ones(shape=[filter_height, filter_width, 1, 1])
        wl = tf.convert_to_tensor(weights_left, dtype=tf.float32)
        wm = tf.convert_to_tensor(weights_middle, dtype=tf.float32)
        wr = tf.convert_to_tensor(weights_right, dtype=tf.float32)
        return wl, wm, wr

    def filter(self, img):
        """
        Use weight hat-like filter to filter the image
        :param img:input image
        :return:
        """
        # In order to start from the edge of the image, making border of the image first
        img = tf.cast(img, tf.float32)
        [height, width, _] = img[0].get_shape().as_list()
        [filter_h, filter_w] = self.__ksize
        new_height = height + 3*filter_h
        new_width = width + 3*filter_w
        _image = tf.image.resize_image_with_crop_or_pad(image=img, target_height=new_height,
                                                        target_width=new_width)
        image_left = _image[:, 0:(new_height - 2*filter_h), 0:(new_width - 2*filter_w), :]
        image_middle = _image[:, filter_h:(new_height - filter_h), filter_w:(new_width - filter_w), :]
        image_right = _image[:, 2*filter_h:new_height, 2*filter_w:new_width, :]

        [weights_left, weights_middle, weights_right] = self.__init_filter_weights(self.__ksize)
        filter_image_left = tf.nn.conv2d(input=image_left, filter=weights_left, strides=[1, 1, 1, 1],
                                         padding='SAME', name='left_conv')
        filter_image_middle = tf.nn.conv2d(input=image_middle, filter=weights_middle, strides=[1, 1, 1, 1],
                                           padding='SAME', name='middle_conv')
        filter_image_right = tf.nn.conv2d(input=image_right, filter=weights_right, strides=[1, 1, 1, 1],
                                          padding='SAME', name='right_conv')
        # Crop the useless edge
        filter_image_left = filter_image_left[
                            :,
                            int(0.5 * filter_h):(filter_image_left.get_shape().as_list()[1] - int(0.5 * filter_h)),
                            int(0.5 * filter_w):(filter_image_left.get_shape().as_list()[2] - int(0.5 * filter_w)),
                            :]
        filter_image_middle = filter_image_middle[
                            :,
                            int(0.5 * filter_h):(filter_image_middle.get_shape().as_list()[1] - int(0.5 * filter_h)),
                            int(0.5 * filter_w):(filter_image_middle.get_shape().as_list()[2] - int(0.5 * filter_w)),
                            :]
        filter_image_right = filter_image_right[
                            :,
                            int(0.5 * filter_h):(filter_image_right.get_shape().as_list()[1] - int(0.5 * filter_h)),
                            int(0.5 * filter_w):(filter_image_right.get_shape().as_list()[2] - int(0.5 * filter_w)),
                            :]
        # calculate the adaptive weight thresh = 1 only if middle > left and middle > right
        thresh_left_middle = tf.less(filter_image_left, filter_image_middle)
        thresh_right_middle = tf.less(filter_image_right, filter_image_middle)
        adaptive_weight = tf.multiply(tf.cast(thresh_left_middle, tf.float32), tf.cast(thresh_right_middle, tf.float32))

        # calculate the thresh result res = ada_weight*(2*middle - left - right)
        tmp = tf.multiply(2.0, filter_image_middle)
        tmp = tf.subtract(tmp, filter_image_left)
        tmp = tf.subtract(tmp, filter_image_right)
        res = tf.multiply(adaptive_weight, tmp)
        return res
