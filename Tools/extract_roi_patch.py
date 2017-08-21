#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : extract_roi_patch.py
"""
Use weights hat-like filter to filter the top view image and extract the roi patch and its' corresponding roi patch in
front view image and save them into data lane_line which are supposed to be selected by hands later. This function is
mainly used during training time to support the sample selection work
"""
import os
import os.path as ops
import argparse
import numpy as np
import shutil
import sys
import time
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from Extract_line_candidates.extract_candidate import RoiExtractor
from Extract_line_candidates.inverse_perspective_map import PerspectiveTransformer
from Global_Configuration.config import cfg

__START_X = cfg.ROI.TOP_CROP_START_X
__START_Y = cfg.ROI.TOP_CROP_START_Y
__CROP_WIDTH = cfg.ROI.TOP_CROP_WIDTH
__CROP_HEIGHT = cfg.ROI.TOP_CROP_HEIGHT


def __get_rect(rotbox):
    """
    Get the min surrounding box around the rotate box
    :param rotbox: rotate box
    :return:
    """
    x_list = [rotbox[0][0], rotbox[1][0], rotbox[2][0], rotbox[3][0]]
    y_list = [rotbox[0][1], rotbox[1][1], rotbox[2][1], rotbox[3][1]]

    xmin = np.amin(a=np.array(x_list), axis=0)
    xmax = np.amax(a=np.array(x_list), axis=0)
    ymin = np.amin(a=np.array(y_list), axis=0)
    ymax = np.amax(a=np.array(y_list), axis=0)

    rect = np.array([xmin, ymin, xmax, ymax])
    return rect


def __init_folders(top_image_dir, fv_image_dir, top_rois_dir, fv_rois_dir):
    """
    Check if the top image dir and fv image dir exist. Delete the existed top rois dir and fv rois dir
    :param top_image_dir:
    :param fv_image_dir:
    :param top_rois_dir:
    :param fv_rois_dir:
    :return:
    """
    if not ops.exists(top_image_dir) or not ops.exists(fv_image_dir):
        raise ValueError('Folder {:s} or {:s} doesn\'t exist'.format(top_image_dir, fv_image_dir))
    if not ops.exists(top_rois_dir):
        os.makedirs(top_rois_dir)
    if not ops.exists(fv_rois_dir):
        os.makedirs(fv_rois_dir)

    if ops.exists(top_rois_dir) and os.listdir(top_rois_dir):  # old top rois dir contains rois delete them
        shutil.rmtree(top_rois_dir)
        os.makedirs(top_rois_dir)
    if ops.exists(fv_rois_dir) and os.listdir(fv_rois_dir):
        shutil.rmtree(fv_rois_dir)
        os.makedirs(fv_rois_dir)

    if not ops.exists(top_rois_dir):
        os.makedirs(top_rois_dir)
    if not ops.exists(fv_rois_dir):
        os.makedirs(fv_rois_dir)
    print('Folders initialization complete!')
    return


def extract_and_save_roi_patch(top_image_dir, fv_image_dir, top_view_roi_save_path, front_view_roi_save_path):
    """
    extract roi patch from top view image and save the roi patch and its' corresponding front view roi patch
    :param top_image_dir: the path where you store the top view image
    :param fv_image_dir: the path where you store the front view image
    :param top_view_roi_save_path: the path where you store the top view roi patch
    :param front_view_roi_save_path: the path where you store the front view roi patch
    :return:
    """
    __init_folders(top_image_dir=top_image_dir, fv_image_dir=fv_image_dir,
                   top_rois_dir=top_view_roi_save_path, fv_rois_dir=front_view_roi_save_path)

    extractor = RoiExtractor(_cfg=cfg)
    res_info = extractor.extract_all(top_image_dir)

    transformer = PerspectiveTransformer(_cfg=cfg)

    extract_count = 0
    for image_id, info in res_info.items():
        # read top view image
        top_image_id = image_id
        top_image = cv2.imread(os.path.join(top_image_dir, top_image_id), cv2.IMREAD_UNCHANGED)
        # read front view image
        fv_image_id = image_id.replace('top', 'fv')
        fv_image = cv2.imread(os.path.join(fv_image_dir, fv_image_id), cv2.IMREAD_UNCHANGED)

        # get roi information from top view image
        rrect_list = info['rrect']

        for index, rrect in enumerate(rrect_list):
            rotbox = np.int0(cv2.boxPoints(rrect))
            # get the min rect surrounding the rotate rect
            top_minrect = __get_rect(rotbox=rotbox)
            # get the top roi
            top_roi = top_image[top_minrect[1]:top_minrect[3], top_minrect[0]:top_minrect[2], :]
            # top view roi save name constructed as xxxx_top_index_ltx_lty_rbx_rby.jpg to reserve the position
            # information in the top view image
            top_roi_save_id = '{:s}_{:d}.jpg'.format(top_image_id[:-4], index)
            top_roi_save_id = os.path.join(top_view_roi_save_path, top_roi_save_id)
            cv2.imwrite(top_roi_save_id, top_roi)

            for j in range(4):
                # remap the rotate rect position from the cropped top view image to the origin top view image which is
                # correspond to the front view image. The cropped top view image is cropped from the origin top view
                # image at position [START_X, START_Y] with [CROP_WIDTH, CROP_HEIGHT] width and height
                rotbox[j] = np.add(rotbox[j], [__START_X, __START_Y])
                # remap the position from top view image to front view image
                rotbox[j] = transformer.perspective_point(rotbox[j])
            # get the min surrounding rect of the rotate rect
            fv_minrect = __get_rect(rotbox)
            fv_roi = fv_image[fv_minrect[1]:fv_minrect[3], fv_minrect[0]:fv_minrect[2], :]
            # fv roi image id was constructed in the same way as top roi image id
            fv_roi_save_id = '{:s}_{:d}.jpg'.format(fv_image_id[:-4], index)
            fv_roi_save_id = os.path.join(front_view_roi_save_path, fv_roi_save_id)
            cv2.imwrite(fv_roi_save_id, fv_roi)
        extract_count += 1
        sys.stdout.write('\r>>Extracting rois {:d}/{:d} {:s}'.format(extract_count, len(res_info), image_id))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_image_dir', type=str, help='The dir where you store the cropped top view images')
    parser.add_argument('--fv_image_dir', type=str, help='The dir where you store the front view images')
    parser.add_argument('--top_rois_dir', type=str, help='The dir where you store the extracted top view rois')
    parser.add_argument('--fv_rois_dir', type=str, help='The dir where you store the extracted front view rois')

    args = parser.parse_args()

    t_start = time.time()
    extract_and_save_roi_patch(top_image_dir=args.top_image_dir, fv_image_dir=args.fv_image_dir,
                               top_view_roi_save_path=args.top_rois_dir, front_view_roi_save_path=args.fv_rois_dir)
    print('Extracting rois complete cost time {:5f}s'.format(time.time() - t_start))
