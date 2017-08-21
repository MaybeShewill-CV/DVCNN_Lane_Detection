#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : config.py
"""
Set some global config
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
# from config import cfg
cfg = __C

# Extract lane line candidates roi options
__C.ROI = edict()

# Set perspective transform control points
__C.ROI.CPT_FV = [(98, 701), (770, 701), (291, 541), (645, 541)]
__C.ROI.CPT_TOP = [(425, 701), (525, 701), (425, 600), (525, 600)]

# Set perspective transform image size (width, height)
__C.ROI.WARPED_SIZE = (1000, 700)

# Set the crop roi start_x, start_y, crop_width, crop_height. Since some parts of the perspective transformed image are
# invalid we are supposed crop a valid sub image of the whole transformed image and these parameters are the positions
# of the sub image
__C.ROI.TOP_CROP_START_X = 325
__C.ROI.TOP_CROP_START_Y = 327
__C.ROI.TOP_CROP_WIDTH = 325
__C.ROI.TOP_CROP_HEIGHT = 325

# Train options
__C.TRAIN = edict()

# Set the weighted hat like filter window width
__C.TRAIN.HAT_LIKE_FILTER_WINDOW_WIDTH = 3
# Set the weighted hat like filter window height
__C.TRAIN.HAT_LIKE_FILTER_WINDOW_HEIGHT = 7
# Set the DVCNN training epochs
__C.TRAIN.EPOCHS = 6000
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the test display step during training process
__C.TRAIN.TEST_DISPLAY_STEP = 100
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.85
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = False
# Set the DVCNN training batch size
__C.TRAIN.BATCH_SIZE = 64
# Set the DVCNN validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 396
# Set the learning rate decay steps
__C.TRAIN.LR_DECAY_STEPS = 2000
# Set the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.1
# Set the L2 regularization decay rate
__C.TRAIN.L2_DECAY_RATE = 0.0001
# Set the horizon flip data augmentation method
__C.TRAIN.USE_HORIZON_FLIP = True
# Set the vertical flip data augmentation method
__C.TRAIN.USE_VERTICAL_FLIP = True
# Set the random crop data augmentation method
__C.TRAIN.USE_RANDOM_CROP = False
__C.TRAIN.RANDOM_CROP_VALUE = [300, 300, 3]
# Set the random brightness data augmentation method
__C.TRAIN.USE_RANDOM_BRIGHTNESS = True
__C.TRAIN.RANDOM_BRIGHTNESS_VALUE = 100  # you can check the preprocess.py scripts to learn the way how it's implemented
# Set the random contrast data augmentation method
__C.TRAIN.USE_RANDOM_CONTRAST = True
__C.TRAIN.RANDOM_CONTRAST_LOWER_VALUE = 0.4
__C.TRAIN.RANDOM_CONTRAST_HIGHER_VALUE = 0.6
# Set the std normalization data augmentation method
__C.TRAIN.USE_STD_NORMALIZATION = False
# Set the min max normalization data augmentation method
__C.TRAIN.USE_MINMAX_NORMALIZATION = False
# Set the central normalization data augmentation method
__C.TRAIN.USE_CENTRAL_NORMALIZATION = False
__C.TRAIN.CENTRAL_NORMALIZATION_VALUE = [103.939, 116.779, 123.68]

# Test options
__C.TEST = edict()
# Set the weighted hat like filter window width
__C.TEST.HAT_LIKE_FILTER_WINDOW_WIDTH = 3
# Set the weighted hat like filter window height
__C.TEST.HAT_LIKE_FILTER_WINDOW_HEIGHT = 7
# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.85
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = False

