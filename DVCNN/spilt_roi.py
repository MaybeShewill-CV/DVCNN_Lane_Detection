"""
Spilt the front view rois into some shares in order to select the positive and negative training samples more easily
"""
import os
import numpy as np
import shutil


def match_str(str1, str2):
    """
    if str1 contains str2 return true else return false
    :param str1:
    :param str2:
    :return:
    """
    if str2 in str1:
        return True
    else:
        return False


def find_substr_itimes(_str, _substr, i):
    """
    find the location of the substr appered i times in str
    :param _str:
    :param _substr:
    :param i:
    :return:
    """
    count = 0
    while i > 0:
        index = _str.find(_substr)
        if index == -1:
            return -1
        else:
            _str = _str[index+1:]
            i -= 1
            count = count + index + 1
    return count - 1

# fv_src_dir = '/home/baidu/Road_Center_Line_DataBase/lane_line/front_view'
# top_src_dir = '/home/baidu/Road_Center_Line_DataBase/lane_line/top_view'
#
# fv_image_id_list = []
# top_image_id_list = []
#
# for parents, dirnames, filenames in os.walk(fv_src_dir):
#     for filename in filenames:
#         fv_image_id_list.append(filename)
#
# for _parents, _dirnames, filenames in os.walk(top_src_dir):
#     for filename in filenames:
#         top_image_id_list.append(filename)
#
# sample_nums = len(fv_image_id_list)
# index = np.random.permutation(sample_nums)
#
# share = 100
# batch_size_per_share = int(sample_nums / share)
#
# for share_index in range(share):
#     if not os.path.exists('/home/baidu/Road_Center_Line_DataBase/lane_line/fv_share_{:d}'.format(share_index)):
#         os.makedirs('/home/baidu/Road_Center_Line_DataBase/lane_line/fv_share_{:d}'.format(share_index))
#
#     index_start = share_index*batch_size_per_share
#     index_end = share_index*batch_size_per_share + batch_size_per_share
#     if index_end > sample_nums:
#         index_end = sample_nums - 1
#     index_per_share = index[index_start:index_end]
#     for index_tmp in index_per_share:
#         fv_image_id = fv_image_id_list[index_tmp]
#         image_id_prefix = fv_image_id[:find_substr_itimes(fv_image_id, '_', 3)]
#         image_id_prefix = image_id_prefix.replace('fv', 'top')
#         top_image_id = ''
#         # top_image_id = [s for s in top_image_id_list if image_id_prefix+'_' in s]
#         # top_image_id = top_image_id[0]
#         for top_image_id_tmp in top_image_id_list:
#             if match_str(top_image_id_tmp, image_id_prefix):
#                 top_image_id = top_image_id_tmp
#                 # top_image_id_list.remove(top_image_id_tmp)
#                 break
#         if top_image_id == '':
#             raise ValueError('Can\'t find {:s} in top image file list'.format(image_id_prefix))
#         old_file_name = os.path.join(fv_src_dir, fv_image_id)
#         if not os.path.isfile(old_file_name):
#             raise ValueError('{:s} doesn\'t exist'.format(old_file_name))
#         new_file_name = os.path.join('/home/baidu/Road_Center_Line_DataBase/lane_line/fv_share_{:d}'
#                                      .format(share_index), fv_image_id)
#         shutil.copy(old_file_name, new_file_name)
#     print('Process share {:d} done'.format(share_index))


