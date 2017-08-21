#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : preprocess.py
"""
Define the image preprocessor class which is mainly used for data augmentation
"""
import tensorflow as tf


class Preprocessor(object):
    def __init__(self):
        self.__augmentation_map = self.__init_augmentation_map()
        pass

    def __str__(self):
        return 'Image Preprocessor object'

    def __init_augmentation_map(self):
        aug_dict = dict()
        aug_dict['whiten'] = self.__augmentation_centralization
        aug_dict['flip_horizon'] = self.__augmentation_random_flip_horizon
        aug_dict['flip_vertical'] = self.__augmentation_random_filp_vertical
        aug_dict['random_crop'] = self.__augmentation_random_crop
        aug_dict['random_brightness'] = self.__augmentation_random_brightness
        aug_dict['random_contrast'] = self.__augmentation_random_contrast
        aug_dict['std_normalization'] = self.__augmentation_std_normalization
        aug_dict['minmax_normalization'] = self.__augmentation_minmax_normalization
        return aug_dict

    @staticmethod
    def __augmentation_random_crop(img, **kwargs):
        """
        random crop image
        :param img: image_tensor
        :param crop_size: crop_size_list
        :return: augmented_image_tensor
        """
        if type(kwargs['crop_size']) != list:
            raise TypeError('crop_size must be list [crop_height, crop_width, channels]')
        if kwargs['crop_size'][0] < 0 or kwargs['crop_size'][1] < 0:
            raise ValueError('Crop size must be bigger than 0 and smaller than the origin image size')
        # Since the tf.random_crop op doesn't support 4-D Tensor with batch, so use tf.map_fn() to operate on each
        # element
        result = tf.map_fn(lambda image: tf.random_crop(value=image, size=kwargs['crop_size']), img)
        return result

    @staticmethod
    def __augmentation_random_flip_horizon(img, **kwargs):
        """
        flip image horizonlly
        :param img: image_tensor
        :return: augmented_image_tensor
        """
        result = tf.map_fn(lambda image: tf.image.random_flip_left_right(image=image), img)
        return result

    @staticmethod
    def __augmentation_random_filp_vertical(img, **kwargs):
        """
        flip image vertically
        :param img: image tensor
        :return: augmented_image_tensor
        """
        result = tf.map_fn(lambda image: tf.image.random_flip_up_down(image=image), img)
        return result

    @staticmethod
    def __augmentation_random_brightness(img, **kwargs):
        """
        random add brightness noise to image and the brightness varies from [-brightess, brightness)
        :param img: origin image tensor
        :param kwargs['brightness']: brightness noise to be added brightness varies from [-brightness, brightness)
        :return: augmented_image_tensor
        """
        result = tf.map_fn(lambda image: tf.image.random_brightness(image=image, max_delta=kwargs['brightness']), img)
        return result

    @staticmethod
    def __augmentation_random_contrast(img, **kwargs):  # lower_factor, upper_factor):
        """
        randomly change the contrast of the image, change factor constrast_factor varies from
        [lower_factor, upper_factor].For each channel, this Op computes the mean of the image pixels in the channel and
        then adjusts each component x of each pixel to (x - mean) * contrast_factor + mean
        :param image: image tensor
        :param lower_factor: lowest constrast factor
        :param upper_factor: uppest constrast factor
        :return: augmented_image_tensor
        """
        result = tf.map_fn(lambda image: tf.image.random_contrast(image=image, lower=kwargs['lower_factor'],
                                                                  upper=kwargs['upper_factor']), img)
        return result

    @staticmethod
    def __augmentation_std_normalization(img, **kwargs):
        """
        Subtract off the mean and divide by the variance of the pixels.(std normalization)
        :param img: origin image tensor
        :return: augmented_image_tensor
        """
        result = tf.map_fn(lambda image: tf.image.per_image_standardization(image=image), img)
        return result

    @staticmethod
    def __augmentation_minmax_normalization(img, **kwargs):
        """
        op: use (pixel - min) / (max - min) to do the normalization
        :param img: image_tensor
        :return: augmented_image_tensor
        """

        def __min_max_norm(image_single):
            pixel_max_tensor = tf.reduce_max(input_tensor=image_single, reduction_indices=[0, 1])
            pixel_min_tensor = tf.reduce_min(input_tensor=image_single, reduction_indices=[0, 1])
            image_single = tf.divide(tf.subtract(image_single, pixel_min_tensor),
                                     tf.subtract(pixel_max_tensor, pixel_min_tensor))
            return image_single

        result = tf.map_fn(lambda image: __min_max_norm(image_single=image), img)
        return result

    @staticmethod
    def __augmentation_centralization(img, **kwargs):
        """
        Image whiten process new_value = origin_value - center_value
        :param img: origin image
        :param center_value: value used to centeralization eg. for ImageNet [104, 117, 123]
        :return:
        """

        def __centralization(image_single):
            mean_value = tf.constant(value=kwargs['center_value'], dtype=tf.float32, shape=[3], name='Image_Mean_Value')
            return tf.subtract(image_single, mean_value)

        result = tf.map_fn(lambda image: __centralization(image_single=image), img)
        return result

    def augment_image(self, image, function_flag, function_params):
        """
        Do data augmentation work
        :param function_flag: refer to which data augmentation to use
        :param function_params: function params
        :param image: input image tensor
        :return:
        """
        aug_method = self.__augmentation_map[function_flag]
        return aug_method(image, **function_params)
