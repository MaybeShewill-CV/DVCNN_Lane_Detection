#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : extract_roi.py
"""
Given the top view image, return the roi list extracted by the weight hat like filter and thresholding function
"""
import cv2
import numpy as np
import tensorflow as tf
try:
    from cv2 import cv2
except ImportError:
    pass

from Extract_line_candidates import filter_util
from Extract_line_candidates import binarized_filter_result


class RoiExtractorSingle(object):
    """
    Extractor single image lane candidate rois used during interfacing process
    """
    def __init__(self, _cfg):
        self.__cfg = _cfg

    def __whatlike_filter_image(self, image):
        """
        Use weight hat like filter filter the single image
        :param image:
        :return:
        """
        if image is None:
            raise ValueError('Image data is invalid')
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        assert image.shape[0] == image.shape[1] == self.__cfg.ROI.TOP_CROP_HEIGHT

        # initialize the weight hat like filter
        whatlikefilter = filter_util.WHatLikeFilter([self.__cfg.TEST.HAT_LIKE_FILTER_WINDOW_HEIGHT,
                                                     self.__cfg.TEST.HAT_LIKE_FILTER_WINDOW_WIDTH])

        # set the input tensor
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, self.__cfg.ROI.TOP_CROP_WIDTH,
                                                               self.__cfg.ROI.TOP_CROP_WIDTH, 1], name='Input_Image')
        input_image = image[np.newaxis, :, :, np.newaxis]

        # set sess config
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.__cfg.TEST.GPU_MEMORY_FRACTION
        config.gpu_options.allow_growth = self.__cfg.TEST.TF_ALLOW_GROWTH

        sess = tf.Session(config=config)

        with sess.as_default():

            init = tf.global_variables_initializer()
            sess.run(init)

            filter_result = sess.run(whatlikefilter.filter(img=input_tensor), feed_dict={input_tensor: input_image})

        return filter_result

    def extract_roi_candidates(self, image):
        """
        extract the candidate roi of the top view image through weight hat like filter and thresholding
        :param image:
        :return:
        """
        if image is None:
            raise ValueError('Image data is invalid')

        # apply the weight hat like filter
        filtered_image = self.__whatlike_filter_image(image=image)

        # apply OTSU threshold and components analysis function to extract the candidates rois
        filterbinarizor = binarized_filter_result.FilterBinarizer(_cfg=self.__cfg)
        roi_pairs, thresh_image = filterbinarizor.binarized_whatlike_filtered_image(img=filtered_image[0])

        return roi_pairs, filtered_image[0], thresh_image
