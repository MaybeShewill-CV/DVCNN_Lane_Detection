#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : test_dvcnn.py
"""
Test the DVCNN network
"""
import tensorflow as tf
import sys
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
import argparse
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from DVCNN import data_provider, dvcnn_model


def test_net(model_path, weights_path, lane_dir, non_lane_dir):
    """
    Test DVCNN network
    :param model_path: DVCNN json model path
    :param weights_path: DVCNN network weights file path(ckpt file)
    :param lane_dir: lane line sample dir where contains top_view and front_view folder
    :param non_lane_dir: non lane line sample dir where contains top_view and front_view folder
    :return:
    """
    # provide the data for testing
    provider = data_provider.DataProvider(lane_dir=lane_dir, not_lane_dir=non_lane_dir)

    test_top_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='top_input')
    test_front_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='front_input')
    test_label_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='label_input')

    test_batch_data = provider.next_batch(batch_size=396)
    test_top_input = []
    test_front_input = []
    test_label = []
    test_front_filename = []
    for index, test_data in enumerate(test_batch_data):
        top_file_name = test_data[0]
        front_file_name = test_data[1]
        label = test_data[2]
        top_image = cv2.imread(top_file_name, cv2.IMREAD_UNCHANGED)
        if top_image is None:
            continue
        top_image = cv2.resize(src=top_image, dsize=(64, 64))
        front_image = cv2.imread(front_file_name, cv2.IMREAD_UNCHANGED)
        if front_image is None:
            continue
        front_image = cv2.resize(src=front_image, dsize=(128, 128))
        test_top_input.append(top_image)
        test_front_input.append(front_image)
        test_label.append(label)
        test_front_filename.append(front_file_name)

    test_label_input = test_label.copy()
    for kk in range(len(test_label_input)):
        if test_label[kk] == 1:
            test_label_input[kk] = [0, 1]
        else:
            test_label_input[kk] = [1, 0]

    # Set the dvcnn builder
    dvcnn = dvcnn_model.DVCNNBuilder(json_model_path=model_path)

    dvcnn_out = dvcnn.build_dvcnn(top_view_input=test_top_input_tensor,
                                  front_view_input=test_front_input_tensor)

    preds = tf.argmax(tf.nn.softmax(dvcnn_out), 1)

    lane_sample_nums = np.count_nonzero(test_label)
    non_lane_sample_nums = len(test_label) - lane_sample_nums

    saver = tf.train.Saver()

    sess = tf.Session()

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        prediction, fv_image_list, top_image_list, gt_label_list = sess.run(
            [preds, test_front_input_tensor, test_top_input_tensor, test_label_input_tensor],
            feed_dict={test_front_input_tensor: test_front_input, test_top_input_tensor: test_top_input,
                       test_label_input_tensor: test_label_input})

        diff = prediction - test_label
        correct_prediction = np.count_nonzero(diff == 0)
        accuracy = correct_prediction / len(test_label)
        print('******Image File ID****** ***GT Label*** ***Prediction label***')
        for index, fv_image in enumerate(fv_image_list):
            file_id = os.path.split(test_front_filename[index])[1]
            print('***{:s}*** ***  {:d}  *** ***  {:d}  ***'.format(file_id, np.argmax(gt_label_list[index], axis=0),
                                                                    prediction[index]))
            # plt.figure('Front View Image')
            # plt.imshow(np.uint8(fv_image[:, :, (2, 1, 0)]))
            # plt.figure('Top View Image')
            # plt.imshow(np.uint8(top_image_list[index][:, :, (2, 1, 0)]))
            # plt.show()
        print('Total test sample is {:d} lane sample nums: {:d} non lane samples nums: {:d}'.
              format(len(test_label), lane_sample_nums, non_lane_sample_nums))
        print('Predicts {:d} images {:d} is correct accuracy is {:4f}'.format(len(test_label), correct_prediction,
                                                                              accuracy))
    return


def test_net_lane(model_path, weights_path, lane_dir, non_lane_dir):
    """
    Select the lane sample which the detector think it is a lane sample
    :param model_path:
    :param weights_path:
    :param lane_dir:
    :param non_lane_dir:
    :return:
    """
    # provide the data for testing
    provider = data_provider.DataProvider(lane_dir=lane_dir, not_lane_dir=non_lane_dir)

    test_top_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='top_input')
    test_front_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='front_input')

    # Set the dvcnn builder
    dvcnn = dvcnn_model.DVCNNBuilder(json_model_path=model_path)

    dvcnn_out = dvcnn.build_dvcnn(top_view_input=test_top_input_tensor, front_view_input=test_front_input_tensor)
    preds = tf.argmax(tf.nn.softmax(dvcnn_out), 1)

    result = []
    result_file = open('DVCNN/lane_result.txt', 'w')

    saver = tf.train.Saver()

    # configuration
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = False

    sess = tf.Session(config=config)

    loop_times = 128
    batch_size = 1000

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        for loops in range(loop_times):
            print('******Start epoch {:d} process {:d}-{:d}/{:d}******'.format(loops, loops*batch_size,
                                                                               loops*batch_size+batch_size,
                                                                               loop_times*batch_size))
            t_start = time.time()
            test_batch_data = provider.next_batch(batch_size=batch_size)
            test_top_input = []
            test_front_input = []
            test_front_filename = []
            for index, test_data in enumerate(test_batch_data):
                top_file_name = test_data[0]
                front_file_name = test_data[1]
                top_image = cv2.imread(top_file_name, cv2.IMREAD_UNCHANGED)
                if top_image is None:
                    continue
                top_image = cv2.resize(src=top_image, dsize=(64, 64))
                front_image = cv2.imread(front_file_name, cv2.IMREAD_UNCHANGED)
                if front_image is None:
                    continue
                front_image = cv2.resize(src=front_image, dsize=(128, 128))
                test_top_input.append(top_image)
                test_front_input.append(front_image)
                test_front_filename.append(front_file_name)
            print('Complete data preparation cost time: {:6f}s'.format(time.time() - t_start))

            t_start = time.time()
            predictions = sess.run(preds, feed_dict={test_front_input_tensor: test_front_input,
                                                     test_top_input_tensor: test_top_input})
            print('Complete dvcnn prediction cost time: {:6f}s'.format(time.time() - t_start))

            t_start = time.time()
            for index, prediction in enumerate(predictions):
                [_, file_id] = os.path.split(test_front_filename[index])
                if prediction == 1:
                    result.append(test_front_filename[index])
                sys.stdout.write('\rPredicts {:d}/{:d} {:s} label: {:d}'.format(loops*batch_size + index,
                                                                                loop_times*batch_size,
                                                                                file_id, prediction))
                time.sleep(0.0000001)
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
            print('Complete analysis the predictions cost time: {:6f}s'.format(time.time() - t_start))

    t_start = time.time()
    result = list(set(result))  # merge the same image name
    for filename in result:
        result_file.write(filename + '\n')
    print('Complete writing down the lane file name cost time: {:6f}'.format(time.time() - t_start))
    return


def select_lane_fv_sample(lane_result_file):
    """
    According the lane result text file to select the front view lane sample result
    :param lane_result_file:
    :return:
    """
    if not os.path.exists('/home/baidu/DataBase/Road_Center_Line_DataBase/'
                          'DVCNN_SAMPLE_TEST/lane_line/front_view_select'):
        os.makedirs('/home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE_TEST/lane_line/front_view_select')

    file = open(lane_result_file, 'r')
    total_count = len(file.readlines())
    file.close()
    with open(lane_result_file, 'r') as file:
        for index, filename in enumerate(file.readlines()):
            filename = filename[0:-1]
            new_filename = filename.replace('front_view', 'front_view_select')
            shutil.copyfile(filename, new_filename)
            sys.stdout.write('\r>>Copy {:d}/{:d} {:s}'.format(index+1, total_count, os.path.split(new_filename)[1]))
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Where you save the DVCNN model definition json file')
    parser.add_argument('--weights_path', type=str, help='Where you save the DVCNN weights file')
    parser.add_argument('--lane_line_dir', type=str, help='Where you store the lane line samples')
    parser.add_argument('--non_lane_line_dir', type=str, help='Where you store the non lane line samples')

    args = parser.parse_args()

    # test_net_lane(model_path=args.model_path, weights_path=args.weights_path, lane_dir=args.lane_line_dir,
    #               non_lane_dir=args.non_lane_line_dir)
    # select_lane_fv_sample(lane_result_file='DVCNN/lane_result.txt')
    test_net(model_path=args.model_path, weights_path=args.weights_path, lane_dir=args.lane_line_dir,
             non_lane_dir=args.non_lane_line_dir)
