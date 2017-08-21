#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : inverse_perspective_map.py
"""
Some perspective related function. The control points are selected by hands
"""
import numpy as np
import argparse
import os
import sys
import time
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from Global_Configuration import config


class PerspectiveTransformer(object):
    def __init__(self, _cfg):
        """
        Set the control point of front view and top view image eg.
        __C.ROI.CPT_FV = [(98, 701), (770, 701), (291, 541), (645, 541)]
        __C.ROI.CPT_TOP = [(425, 701), (525, 701), (425, 600), (525, 600)]
        :param _cfg:
        """
        self.__front_view_ctrl_point = _cfg.ROI.CPT_FV
        self.__top_view_ctrl_point = _cfg.ROI.CPT_TOP
        self.__warped_size = _cfg.ROI.WARPED_SIZE

    def inverse_perspective_map(self, image):
        """
        Convert front view image to top view image
        :param image:
        :return:
        """
        fv_ctrl_point = np.array(self.__front_view_ctrl_point).astype(dtype=np.float32)
        top_ctrl_point = np.array(self.__top_view_ctrl_point).astype(dtype=np.float32)

        warp_transform = cv2.getPerspectiveTransform(src=fv_ctrl_point, dst=top_ctrl_point)
        warped_image = cv2.warpPerspective(src=image, M=warp_transform, dsize=self.__warped_size)
        return warped_image

    def perspective_map(self, image):
        """
        Convert top view image to front view image
        :param image:
        :return:
        """
        top_ctrl_point = np.array(self.__top_view_ctrl_point).astype(dtype=np.float32)
        fv_ctrl_point = np.array(self.__front_view_ctrl_point).astype(dtype=np.float32)

        warp_transform = cv2.getPerspectiveTransform(src=top_ctrl_point, dst=fv_ctrl_point)
        warped_image = cv2.warpPerspective(src=image, M=warp_transform, dsize=self.__warped_size)
        return warped_image

    def inverse_perspective_point(self, pt1):
        """
        map point in front view image into top view image
        :param pt1:
        :return: pt2 [x, y]
        """
        fv_ctrl_point = np.array(self.__front_view_ctrl_point).astype(dtype=np.float32)
        top_ctrl_point = np.array(self.__top_view_ctrl_point).astype(dtype=np.float32)

        warp_transform = cv2.getPerspectiveTransform(src=fv_ctrl_point, dst=top_ctrl_point)
        pt_warp = cv2.perspectiveTransform(src=pt1, m=warp_transform)
        return pt_warp[0, 0, :]

    def perspective_point(self, pt1):
        """
        map point in top view image into front view image
        :param pt1:
        :return: pt2 [x, y]
        """
        pt1 = np.array([[pt1]], dtype=np.float32)

        top_ctrl_point = np.array(self.__top_view_ctrl_point).astype(dtype=np.float32)
        fv_ctrl_point = np.array(self.__front_view_ctrl_point).astype(dtype=np.float32)

        warp_transform = cv2.getPerspectiveTransform(src=top_ctrl_point, dst=fv_ctrl_point)
        pt_warp = cv2.perspectiveTransform(src=pt1, m=warp_transform)
        return pt_warp[0, 0, :]

    def inverse_perspective_map_fvfiles(self, fv_image_src, top_image_path):
        """
        Inverse front view images to top view images
        :param fv_image_src: source front view images
        :param top_image_path: path to store inverse perspective mapped top view images
        :return:
        """
        if not os.path.isdir(fv_image_src):
            raise ValueError('Folder {:s} doesn\'t exist'.format(fv_image_src))
        if not os.path.exists(top_image_path):
            os.makedirs(top_image_path)

        for parents, _, filenames in os.walk(fv_image_src):
            for index, filename in enumerate(filenames):
                fv_img_id = os.path.join(parents, filename)
                fv_img = cv2.imread(fv_img_id, cv2.IMREAD_UNCHANGED)
                top_image = self.inverse_perspective_map(image=fv_img)
                top_image_save_path = os.path.join(top_image_path, filename.replace('fv', 'top'))
                cv2.imwrite(top_image_save_path, top_image)
                sys.stdout.write('\r>>Map {:d}/{:d} {:s}'.format(index+1, len(filenames),
                                                                 os.path.split(top_image_save_path)[1]))
                sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fv_image_dir', type=str, help='The path you store the front view image')
    parser.add_argument('top_image_dir', type=str, help='The path you store the inverse perspective top view images')

    args = parser.parse_args()

    t_start = time.time()
    transformer = PerspectiveTransformer(_cfg=config.cfg)
    transformer.inverse_perspective_map_fvfiles(fv_image_src=args.fv_image_dir, top_image_path=args.top_image_dir)
    print('Inverse perspective done cost time {:5f}s'.format(time.time() - t_start))
