"""
Global optimization according the paper "Accurate and Robust Lane Detection based on Dual-View Convolutional
Neutral Network" formulation (4)
1.First through calculating the horizon diff between every two rois to merge the rois
2.Calculate the score of every merged roi by the w_len and w_amount
3.Remain the merged roi with the highest score
"""
import itertools
import math

import numpy as np

from Global_Configuration.imdb import Roidb


class Optimizer(object):
    def __init__(self, roidb_pair_list):
        """
        Optimizer initialization
        :param roidb_pair_list: each element in roidb_pair_list is a roidb pair (top_roidb, fv_roidb)
        """
        self.__rois = np.array(roidb_pair_list)[:, 0]  # only remain the top roi to do the optimization
        self.__rois_nums = len(roidb_pair_list)
        self.__index_combination = self.__calculate_all_combinations()

    def __calculate_all_combinations(self):
        """
        Calculate all the combination of the roi in self.__rois
        :return: the combination of the index
        """
        index = np.linspace(0, self.__rois_nums-1, self.__rois_nums).astype(np.int32)
        combinations = []
        for i in range(index.shape[0] + 1):
            for subset in itertools.combinations(index, i):
                combinations.append(subset)
        return combinations

    @staticmethod
    def __calculate_wamo(n, sigma=10):
        """
        Calculate the wamo according to the formulation (5)
        :param n: the amount of the lane lines
        :param sigma: the threshold
        :return:
        """
        return math.exp(-math.pow(n, 2) / math.pow(sigma, 2))

    @staticmethod
    def __sigmoid(x):
        """
        Sigmoid function
        :param x:
        :return:
        """
        return 1 / (1 + math.exp(x))

    @staticmethod
    def __activate_softly(x):
        """
        Softly activate x
        :param x:
        :return:
        """
        return 1 / (1 - math.log(x, 100))

    @staticmethod
    def __calculate_wlen_l(s_l, h):
        """
        Calculate the wlen_l according to the softly activate function 1 / (1 - log(x)) which 0 < x < 1
        :param s_l:
        :param h:
        :return:
        """
        return Optimizer.__activate_softly(x=float(s_l / h))

    @staticmethod
    def __calculate_rho_l(rho):
        """
        Calculate the rho score according to the standard that the line with a rho more closer to 90 degree gets more
        high score
        :param rho:
        :return:
        """
        pow_pow = -abs(rho-90) / 90
        return math.pow(math.e, pow_pow)

    @staticmethod
    def __calculate_wlink_ml(rl, rm, rhol, rhom, rmin, rmax, rho):
        """
        Calculate the wlink_ml according to the formulation (7)
        :param rl: the radius of lane l
        :param rm: the radium of lane m
        :param rhol: the angle of lane l
        :param rhom: the angle of lane m
        :param rmin: the min threshold of radius
        :param rmax: the max threshold of radius
        :param rho: the threshold of angle
        :return:
        """
        if abs(rhol - rhom) > rho:
            return float('-inf')
        if abs(rl - rm) > rmin:
            return float('-inf')
        return 0

    @staticmethod
    def __merge_roia_roib(roia, roib):
        """
        Merge roidb a and roidb b
        :param roia:
        :param roib:
        :return:
        """
        contours = np.concatenate((roia.get_roi_contours(), roib.get_roi_contours()), axis=0)
        index = min(roia.get_roi_index(), roib.get_roi_index())
        response_points = np.concatenate((roia.get_roi_response_points(), roib.get_roi_response_points()))
        return Roidb(roi_index=index, roi_contours=contours, roi_response_points=response_points)

    @staticmethod
    def __merge_rois_by_wlink(roidb_list):
        """
        According to the formulation (7) to merge the rois which are supposed to belong to the same lane line
        :param roidb_list:
        :return:
        """
        assert len(roidb_list) > 0
        merged_roidb_list_is_not_stable = True
        merged_roidb_list = roidb_list
        while merged_roidb_list_is_not_stable:
            merged_roidb_list_copy = merged_roidb_list
            merged_roidb_list = []
            merged_times = 0
            for lane_index, lane_l in enumerate(merged_roidb_list_copy):
                rest_roidb_list = merged_roidb_list_copy[lane_index+1:]
                lane_l_has_been_merged = False
                for rest_index, lane_m in enumerate(rest_roidb_list):
                    wlink = Optimizer.__calculate_wlink_ml(rl=lane_l.get_roi_line_radius(),
                                                           rm=lane_m.get_roi_line_radius(),
                                                           rhol=lane_l.get_roi_line_angle(),
                                                           rhom=lane_m.get_roi_line_angle(),
                                                           rmin=10,  # horizon pixel diff should be smaller than 5
                                                           rmax=100,
                                                           rho=15)
                    if wlink == 0:
                        merged_roidb_list.append(Optimizer.__merge_roia_roib(lane_l, lane_m))
                        merged_roidb_list = list(np.concatenate((merged_roidb_list,
                                                                 rest_roidb_list[0:rest_index],
                                                                 rest_roidb_list[rest_index+1:]),
                                                                axis=0))
                        merged_times += 1
                        lane_l_has_been_merged = True
                        break
                if lane_l_has_been_merged:
                    break
                else:
                    merged_roidb_list.append(lane_l)
                    continue
            if merged_times > 0:
                merged_roidb_list_is_not_stable = True
            else:
                merged_roidb_list_is_not_stable = False
        return merged_roidb_list

    def calculate_rois_score(self):
        """
        Calculate the each rois' score according the roi length and angle
        :return:
        """
        merged_roi_list = self.__merge_rois_by_wlink(self.__rois)
        result = []
        for index, roi in enumerate(merged_roi_list):
            lane_length = roi.get_roi_line_length()
            top_image_height = 350.0
            rho = roi.get_roi_line_angle()
            result.append((roi, 0.7*self.__calculate_wlen_l(s_l=lane_length, h=top_image_height) +
                           0.3*self.__calculate_rho_l(rho=rho)))
        return result
