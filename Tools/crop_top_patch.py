#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : crop_top_patch.py
"""
Crop top view image to get a roi image, record the crop start position in order to be able to remap the points in the
roi image to front view image
"""
import os
import sys
import time
import argparse
try:
    from cv2 import cv2
except ImportError:
    pass

from Global_Configuration.config import cfg

START_X = cfg.ROI.TOP_CROP_START_X
START_Y = cfg.ROI.TOP_CROP_START_Y
CROP_WIDTH = cfg.ROI.TOP_CROP_WIDTH
CROP_HEIGHT = cfg.ROI.TOP_CROP_HEIGHT


def crop_image(src, start_x, start_y, width, height):
    [src_height, src_width] = src.shape[:-1]
    assert (start_y + height) < src_height
    assert (start_x + width) < src_width

    if len(src.shape) == 2:  # with 1 channel
        return src[start_y:start_y+height, start_x:start_x+width]
    elif len(src.shape) == 3:
        return src[start_y:start_y+height, start_x:start_x+width, :]
    else:
        raise ValueError('Invalid image shape')


def crop_top_images(top_image_dir, crop_image_save_dir):
    """
    Crop the transformed top images
    :param top_image_dir: the dir of inverse perspective transformed imagse
    :param crop_image_save_dir:
    :return:
    """

    if not os.path.exists(top_image_dir):
        raise ValueError('{:s} doesn\'t exist'.format(top_image_dir))
    if not os.path.exists(crop_image_save_dir):
        os.makedirs(crop_image_save_dir)

    for parents, dirnames, filenames in os.walk(top_image_dir):
        for index, filename in enumerate(filenames):
            top_image_filename = os.path.join(parents, filename)
            top_image = cv2.imread(top_image_filename, cv2.IMREAD_UNCHANGED)
            top_crop_image = crop_image(src=top_image, start_x=START_X, start_y=START_Y, width=CROP_WIDTH,
                                        height=CROP_HEIGHT)
            crop_roi_save_path = os.path.join(crop_image_save_dir, filename)
            cv2.imwrite(crop_roi_save_path, top_crop_image)
            sys.stdout.write('\r>>Map {:d}/{:d} {:s}'.format(index + 1, len(filenames), filename))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('top_image_dir', type=str, help='The dir of inverse perspective transformed top images')
    parser.add_argument('crop_image_dir', type=str, help='The dir of cropped top images')

    args = parser.parse_args()

    t_start = time.time()
    crop_top_images(top_image_dir=args.top_image_dir, crop_image_save_dir=args.crop_image_dir)
    print('Crop complete costs time {:5f}s'.format(time.time() - t_start))
