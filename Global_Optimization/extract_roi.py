"""
Given the top view image, return the roi list extracted by the weight hat like filter and thresholding function
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from Extract_line_candidates.filter_util import WHatLikeFilter
from Extract_line_candidates.binarized_filter_result import binarized_whatlike_filtered_image


def whatlike_filter_image(image):
    """
    Use weight hat like filter filter the single image
    :param image:
    :return:
    """
    if image is None:
        raise ValueError('Image data is invalid')
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    assert image.shape[0] == image.shape[1] == 325

    # initialize the weight hat like filter
    whatlikefilter = WHatLikeFilter([9, 4])

    # set the input tensor
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 325, 325, 1], name='Input_Image')
    input_image = image[np.newaxis, :, :, np.newaxis]

    # set sess config
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = False

    sess = tf.Session(config=config)

    with sess.as_default():

        init = tf.global_variables_initializer()
        sess.run(init)

        filter_result = sess.run(whatlikefilter.filter(img=input_tensor), feed_dict={input_tensor: input_image})

    return filter_result


def extract_roi_candidates(image):
    """
    extract the candidate roi of the top view image through weight hat like filter and thresholding
    :param image:
    :return:
    """
    if image is None:
        raise ValueError('Image data is invalid')

    # apply the weight hat like filter
    filtered_image = whatlike_filter_image(image=image)

    # apply OTSU threshold and components analysis function to extract the candidates rois
    roi_pairs = binarized_whatlike_filtered_image(img=filtered_image[0])

    return roi_pairs, filtered_image[0]
