#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : imdb.py
"""
Construct a roi class to store the candidate information of an image
"""
import math
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from Extract_line_candidates import ransac_fitline


class Roidb(object):
    def __init__(self, roi_index, roi_contours, roi_response_points):
        """
        Construct a roidb classs
        :param roi_index: the index of the roi to identify the roi
        :param roi_contours: the contours of the roi after apply threshold to the result of the hat-like filter image
        """
        self.__roi_index = roi_index
        self.__roi_contours = roi_contours
        self.__roi_response_points = roi_response_points
        if len(self.__roi_contours) == 0:
            # find the bounding box of the contours
            self.__roi_bndbox = [0, 0, 0, 0]
            # fit the line of the contours
            self.__roi_line_param = [0, 0, 0, 0]
            # calculate the line angle
            self.__roi_line_angle = 0
            # calculate the line radius
            self.__roi_line_radius = 0
            # calculate the length of the line
            self.__roi_line_length = 0
            # set the roi scores by the DVCNN classifier
            self.__roi_dvcnn_score = 0
        else:
            # find the bounding box of the contours
            self.__roi_bndbox = self.__boundingRect()
            # fit the line of the contours
            self.__roi_line_param = self.__calculate_line_param()
            # calculate the line angle
            self.__roi_line_angle = self.__calculate_line_angle()
            # calculate the line radius
            self.__roi_line_radius = self.__calculate_line_radius()
            # calculate the length of the line
            self.__roi_line_length = self.__calculate_line_length()
            # set the roi scores by the DVCNN classifier
            self.__roi_dvcnn_score = 0

    def __boundingRect(self):
        """
        Get the bounding box of the contours
        :return:
        """
        return cv2.boundingRect(points=self.__roi_contours)

    def __calculate_line_param(self):
        """
        Using ransac method to calculate the line param ('origin', 'direction')
        :return:
        """
        # param, inliers = ransac_linefit_sklearn(points=self.__roi_contours)
        # param = ransac_linefit_opencv(points=self.__roi_contours)
        ransaclinefitter = ransac_fitline.RansacLineFitter()
        param = ransaclinefitter.ransac_linefit_opencv(points=self.__roi_response_points)
        return param

    def __calculate_line_angle(self):
        """
        Calculate the angle of the line according to the formulation atan(y / x)
        :return:
        """
        assert len(self.__roi_line_param) == 4
        vx = self.__roi_line_param[0]
        vy = self.__roi_line_param[1]
        angle = abs(math.degrees(math.atan2(vy, vx)))
        return angle

    def __calculate_line_radius(self):
        """
        Calculate the radius of the line
        :return:
        """
        assert len(self.__roi_line_param) == 4
        return self.__roi_line_param[2]

    def __calculate_line_length(self):
        """
        Calculate the length of the lane line
        :return:
        """
        return self.__roi_bndbox[3]

    def set_roi_dvcnn_score(self, score):
        self.__roi_dvcnn_score = score

    def get_roi_index(self):
        return self.__roi_index

    def get_roi_contours(self):
        return self.__roi_contours

    def get_roi_bndbox(self):
        return self.__roi_bndbox

    def get_roi_line_param(self):
        return self.__roi_line_param

    def get_roi_line_angle(self):
        return self.__roi_line_angle

    def get_roi_line_radius(self):
        return self.__roi_line_radius

    def get_roi_line_length(self):
        return self.__roi_line_length

    def get_roi_dvcnn_score(self):
        return self.__roi_dvcnn_score

    def get_roi_response_points(self):
        return self.__roi_response_points
