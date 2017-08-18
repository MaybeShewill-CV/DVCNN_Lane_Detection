"""
Make DVCNN training data sets what you are supposed to do is to put the select the lane line and non lane line fv
samples by hand and storing them into folder front_view_lane_line_for_training and front_view_non_lane_line_for_training
"""
import os
import os.path as ops
import numpy as np
import shutil
import sys
import time
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

DVCNN_TRAIN_DATASET_DST_DIR = '/home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE'
DVCNN_TRAIN_DATASET_SRC_DIR = '/home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE_TEST/lane_line'


def initialize_folder():
    """
    Initialize the datasets folder including removing the old folder and making the new folder
    :return:
    """
    # removing the old folders
    old_lane_line_dir = ops.join(DVCNN_TRAIN_DATASET_DST_DIR, 'lane_line')
    old_non_lane_line_dir = ops.join(DVCNN_TRAIN_DATASET_DST_DIR, 'non_lane_line')
    if ops.exists(old_lane_line_dir):
        shutil.rmtree(old_lane_line_dir)
    if ops.exists(old_non_lane_line_dir):
        shutil.rmtree(old_non_lane_line_dir)

    # removing the old top view rois folders
    old_lane_line_top_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'top_view_lane_line_for_training')
    old_non_lane_line_top_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'top_view_non_lane_line_for_training')
    if not ops.isdir(old_lane_line_top_dir) or not ops.isdir(old_non_lane_line_top_dir):
        raise ValueError('{:s} or {:s} doesn\'t exist'.format(old_lane_line_top_dir, old_non_lane_line_top_dir))
    shutil.rmtree(old_lane_line_top_dir)
    shutil.rmtree(old_non_lane_line_top_dir)
    shutil.rmtree(ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'tmp'))

    # making new folders
    new_lane_line_dir = ops.join(DVCNN_TRAIN_DATASET_DST_DIR, 'lane_line')
    lane_line_fv_dir = ops.join(new_lane_line_dir, 'front_view')
    lane_line_top_dir = ops.join(new_lane_line_dir, 'top_view')
    os.makedirs(new_lane_line_dir)
    os.makedirs(lane_line_fv_dir)
    os.makedirs(lane_line_top_dir)

    new_non_lane_line_folder = ops.join(DVCNN_TRAIN_DATASET_DST_DIR, 'non_lane_line')
    non_lane_line_fv_dir = ops.join(new_non_lane_line_folder, 'front_view')
    non_lane_line_top_dir = ops.join(new_non_lane_line_folder, 'top_view')
    os.makedirs(new_non_lane_line_folder)
    os.makedirs(non_lane_line_fv_dir)
    os.makedirs(non_lane_line_top_dir)

    new_lane_line_top_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'top_view_lane_line_for_training')
    new_non_lane_line_top_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'top_view_non_lane_line_for_training')
    os.makedirs(new_lane_line_top_dir)
    os.makedirs(new_non_lane_line_top_dir)
    os.makedirs(ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'tmp'))
    print('Initialize folders done')
    return


def select_lane_line_top_samples():
    """
    Select lane line top samples and restore them into 'top_view_lane_line_for_training' folder
    :return:
    """
    lane_line_fv_samples_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'front_view_lane_line_for_training')
    top_samples_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'top_view')  # All the top samples are stored here
    lane_line_top_sample_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'top_view_lane_line_for_training')
    if not ops.isdir(lane_line_fv_samples_dir):
        raise ValueError('{:s} doesn\'t exist'.format(lane_line_fv_samples_dir))
    if not ops.isdir(top_samples_dir):
        raise ValueError('{:s} doesn\'t exist'.format(top_samples_dir))

    for parents, _, filenames in os.walk(lane_line_fv_samples_dir):
        for index, filename in enumerate(filenames):
            top_file_name = ops.join(top_samples_dir, filename.replace('fv', 'top'))
            if not ops.isfile(top_file_name):
                raise ValueError('{:s} doesn\'t exist'.format(top_file_name))
            fv_file_name = ops.join(parents, filename)
            top_image = cv2.imread(top_file_name, cv2.IMREAD_UNCHANGED)
            fv_image = cv2.imread(fv_file_name, cv2.IMREAD_UNCHANGED)
            if top_image is None or fv_image is None:
                raise ValueError('Image pair {:s} {:s} is not valid'.format(fv_file_name, top_file_name))
            dst_path = ops.join(lane_line_top_sample_dir, filename.replace('fv', 'top'))
            shutil.copyfile(top_file_name, dst_path)
            sys.stdout.write('\r>>Selecting top lane line samples {:d}/{:d} {:s}'.format(index+1, len(filenames),
                                                                                         filename.replace('fv', 'top')))
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    return


def select_non_lane_line_samples():
    """
    Select the non lane line samples. The number of the non lane line samples is the same as lane line samples. The non
    lane line samples are randomly selected from non lane line dir so you must be sure that non lane line samples are
    larger than lane line samples
    :return:
    """
    lane_line_sample_names = os.listdir(ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'front_view_lane_line_for_training'))
    lane_line_sample_nums = len(lane_line_sample_names)

    select_index = np.random.permutation(lane_line_sample_nums)

    non_lane_line_fv_samples_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'front_view_non_lane_line_for_training')
    non_lane_line_top_samples_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'top_view_non_lane_line_for_training')
    non_lane_line_fv_samples_tmp_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'tmp')

    top_samples_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'top_view')  # All the top samples are stored here

    for parents, _, filenames in os.walk(non_lane_line_fv_samples_dir):
        for index, select_id in enumerate(select_index):
            fv_filename = ops.join(parents, filenames[select_id])
            top_filename = ops.join(top_samples_dir, filenames[select_id].replace('fv', 'top'))
            if not ops.isfile(top_filename):
                raise ValueError('{:s} doesn\'t exist'.format(top_filename))
            fv_image = cv2.imread(fv_filename, cv2.IMREAD_UNCHANGED)
            top_image = cv2.imread(top_filename, cv2.IMREAD_UNCHANGED)
            if fv_image is None or top_image is None:
                raise ValueError('Image pair {:s} {:s} is not valid'.format(fv_filename, top_filename))
            top_dst_path = ops.join(non_lane_line_top_samples_dir, filenames[select_id].replace('fv', 'top'))
            fv_dst_path = ops.join(non_lane_line_fv_samples_tmp_dir, filenames[select_id])
            shutil.copyfile(src=top_filename, dst=top_dst_path)
            shutil.copyfile(src=fv_filename, dst=fv_dst_path)
            sys.stdout.write('\r>>Selecting non lane line samples {:d}/{:d} '
                             '{:s}'.format(index+1, lane_line_sample_nums,
                                           filenames[select_id][0:filenames[select_id].find('.')]))
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    return


def copy_samples():
    """
    Copying samples from DVCNN_TRAIN_DATASET_SRC_DIR to DVCNN_TRAIN_DATASET_DST_DIR
    :return:
    """
    lane_line_fv_src_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'front_view_lane_line_for_training')
    lane_line_top_src_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'top_view_lane_line_for_training')
    non_lane_line_fv_src_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'tmp')
    non_lane_line_top_src_dir = ops.join(DVCNN_TRAIN_DATASET_SRC_DIR, 'top_view_non_lane_line_for_training')

    lane_line_fv_dst_dir = ops.join(DVCNN_TRAIN_DATASET_DST_DIR, 'lane_line/front_view')
    lane_line_top_dst_dir = ops.join(DVCNN_TRAIN_DATASET_DST_DIR, 'lane_line/top_view')
    non_lane_line_fv_dst_dir = ops.join(DVCNN_TRAIN_DATASET_DST_DIR, 'non_lane_line/front_view')
    non_lane_line_top_dst_dir = ops.join(DVCNN_TRAIN_DATASET_DST_DIR, 'non_lane_line/top_view')

    for parents, _, filenames in os.walk(lane_line_fv_src_dir):
        for index, filename in enumerate(filenames):
            fv_src_filename = ops.join(parents, filename)
            top_src_filename = ops.join(lane_line_top_src_dir, filename.replace('fv', 'top'))

            fv_dst_filename = ops.join(lane_line_fv_dst_dir, filename)
            top_dst_filename = ops.join(lane_line_top_dst_dir, filename.replace('fv', 'top'))

            shutil.copyfile(src=fv_src_filename, dst=fv_dst_filename)
            shutil.copyfile(src=top_src_filename, dst=top_dst_filename)
            sys.stdout.write('\r>>Copying lane line samples {:d}/{:d} {:s}'.format(index+1, len(filenames),
                                                                                   filename[0:filename.find('.')]))
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

    for parents, _, filenames in os.walk(non_lane_line_fv_src_dir):
        for index, filename in enumerate(filenames):
            fv_src_filename = ops.join(parents, filename)
            top_src_filename = ops.join(non_lane_line_top_src_dir, filename.replace('fv', 'top'))

            fv_dst_filename = ops.join(non_lane_line_fv_dst_dir, filename)
            top_dst_filename = ops.join(non_lane_line_top_dst_dir, filename.replace('fv', 'top'))

            shutil.copyfile(src=fv_src_filename, dst=fv_dst_filename)
            shutil.copyfile(src=top_src_filename, dst=top_dst_filename)
            sys.stdout.write('\r>>Copying non lane line samples {:d}/{:d} {:s}'.format(index + 1, len(filenames),
                                                                                       filename[0:filename.find('.')]))
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    return


def make_new_dataset():
    """
    Making new training dataset mainly copying the lane line front roi and selecting non lane line front roi before
    copying their corresponding top view rois
    :return:
    """
    t_start = time.time()
    # 1.Initialize the folder
    initialize_folder()
    # 2.Select the lane line top samples
    select_lane_line_top_samples()
    # 3.Select the non lane line samples
    select_non_lane_line_samples()
    # 3.Copy the total samples
    copy_samples()
    print('Making DVCNN training datasets complete costs time: {:5f}s'.format(time.time() - t_start))
    return

if __name__ == '__main__':
    make_new_dataset()
