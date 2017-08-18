"""
Data file for training DVCNN should be structured as follows:
    Lane_line_data:
        --top_view:
            --1_lane_top.jpg
            --2_lane_top.jpg
            --......
        --front_view:
            --1_lane_fv.jpg
            --2_lane_fv.jpg
            --......
    Not_lane_line_data:
        --top_view:
            --1_nonlane_top.jpg
            --2_nonlane_top.jpg
            --......
        --front_view:
            --1__nonlane_fv.jpg
            --2__nonlane_fv.jpg
            --......
"""
import os
import numpy as np

from Global_Configuration.utils import find_substr_itimes


class DataProvider(object):
    def __init__(self, lane_dir, not_lane_dir):
        """

        :param lane_dir:where you store the lane picture
        :param not_lane_dir: where you store the not lane picture
        """
        if not os.path.exists(lane_dir):
            raise ValueError('{:s} doesn\'t exist'.format(lane_dir))
        if not os.path.exists(not_lane_dir):
            raise ValueError('{:s} doesn\'t exist'.format(not_lane_dir))

        self.__total_sample = {}
        self.__lane_top_sample_list, self.__lane_front_sample_list = self.__init_lane_sample_list(lane_dir=lane_dir)
        assert len(self.__lane_top_sample_list) == len(self.__lane_front_sample_list)
        self.__nonlane_top_sample_list, self.__nonlane_front_sample_list = \
            self.__init_nonlane_sample_list(nonlane_dir=not_lane_dir)
        assert len(self.__nonlane_top_sample_list) == len(self.__nonlane_front_sample_list)
        self.__sample_nums = len(self.__lane_top_sample_list) + len(self.__nonlane_top_sample_list)
        self.__index = np.random.permutation(self.__sample_nums)
        self.__lane_dir = lane_dir
        self.__non_lane_dir = not_lane_dir

    def __str__(self):
        info = 'Positive samples: {:d} Negative samples: {:d}'.format(len(self.__lane_top_sample_list),
                                                                      len(self.__nonlane_top_sample_list))
        return info

    @staticmethod
    def __init_lane_sample_list(lane_dir):
        """
        Return all the lane patch file name
        :param lane_dir: where you store lane patch file
        :return:
        """
        if not os.path.exists(os.path.join(lane_dir, 'top_view')):
            raise ValueError('{:s} should contain folder top_view'.format(lane_dir))
        if not os.path.exists(os.path.join(lane_dir, 'front_view')):
            raise ValueError('{:s} should contain folder front_view'.format(lane_dir))

        result_top = []
        for parents, dirnames, filenames in os.walk(os.path.join(lane_dir, 'top_view')):
            for filename in filenames:
                tmp = os.path.join(parents, filename)
                result_top.append(tmp)

        result_front = []
        for parents, dirnames, filenames in os.walk(os.path.join(lane_dir, 'front_view')):
            for filename in filenames:
                tmp = os.path.join(parents, filename)
                result_front.append(tmp)
        return result_top, result_front

    @staticmethod
    def __init_nonlane_sample_list(nonlane_dir):
        """
        Return all the non lane patch file name
        :param nonlane_dir: where you store the non lane patch file
        :return:
        """
        if not os.path.exists(os.path.join(nonlane_dir, 'top_view')):
            raise ValueError('{:s} should contain folder top_view'.format(nonlane_dir))
        if not os.path.exists(os.path.join(nonlane_dir, 'front_view')):
            raise ValueError('{:s} should contain folder front_view'.format(nonlane_dir))

        result_top = []
        for parents, dirnames, filenames in os.walk(os.path.join(nonlane_dir, 'top_view')):
            for filename in filenames:
                tmp = os.path.join(parents, filename)
                result_top.append(tmp)

        result_front = []
        for parents, dirnames, filenames in os.walk(os.path.join(nonlane_dir, 'front_view')):
            for filename in filenames:
                tmp = os.path.join(parents, filename)
                result_front.append(tmp)
        return result_top, result_front

    def __init_total_sample_dict(self):
        """
        construct total sample dict {'filename': label} with lane: labeled 1, non lane: labeled 0
        :return:
        """
        # lane patch labeled 1
        for lane_file in self.__lane_top_sample_list:
            self.__total_sample[lane_file] = 1

        # non lane patch labeled 0
        for nonlane_file in self.__nonlane_top_sample_list:
            self.__total_sample[nonlane_file] = 0

    def __find_fv_image_id(self, top_file_name, label):
        """
        According the top file name find it's corresponding fv file name mainly depends on the image_prefix and the
        patch index
        :param top_file_name:
        :param label: used for location the file is lane_line or non_lane_line
        :return:
        """
        [_, top_image_id] = os.path.split(top_file_name)
        top_prefix = top_image_id[:find_substr_itimes(top_image_id, '_', 3)]
        if label == 1:
            fv_file_list = self.__lane_front_sample_list
        else:
            fv_file_list = self.__nonlane_front_sample_list

        for fv_file_name in fv_file_list:
            [_, fv_image_id] = os.path.split(fv_file_name)
            fv_prefix = fv_image_id[:find_substr_itimes(fv_image_id, '_', 3)]
            top_prefix = top_prefix.replace('top', 'fv')
            if fv_prefix == top_prefix:
                return fv_file_name

        raise ValueError('Can\'t find {:s}\'s corresponding image in front view image list'.format(top_file_name))

    def next_batch(self, batch_size):
        """
        Convert dict into a list each item in the list is a dict {filename: label}, then use numpy to generate random
        index to get the shuffle batch
        :return:
        """
        # put dict sample into a list
        self.__init_total_sample_dict()
        sample_list = []
        for patch_name, label in self.__total_sample.items():
            tmp = dict()
            tmp[patch_name] = label
            sample_list.append(tmp)

        if (self.__sample_nums - batch_size - 1) < 0:
            index1 = self.__index[0:self.__sample_nums]
            index2 = self.__index[-(batch_size - self.__sample_nums):]
            if self.__sample_nums == batch_size:
                index = index1
            else:
                index = np.concatenate((index1, index2), 0)
            self.__sample_nums = self.__index.shape[0] - batch_size + self.__sample_nums
        else:
            index = self.__index[self.__sample_nums - batch_size - 1:self.__sample_nums - 1]
            self.__sample_nums -= batch_size

        result = []
        for i in range(index.shape[0]):
            id = index[i]
            top_file_name = list(sample_list[id].keys())[0]
            [top_file_dir, top_file_id] = os.path.split(top_file_name)
            front_file_dir = top_file_dir.replace('top', 'front')
            front_file_id = top_file_id.replace('top', 'fv')
            label = list(sample_list[id].values())[0]
            front_file_name = os.path.join(front_file_dir, front_file_id)
            result.append((top_file_name, front_file_name, label))
        return result
