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
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

import data_provider

from cnn_util import conv2d, batch_norm, activate, max_pool, fully_connect, concat, read_json_model


def build_dvcnn(top_view_input, front_view_input, dvcnn_architecture):
    """
    Build DVCNN net work
    :param top_view_input:
    :param front_view_input:
    :param dvcnn_architecture:
    :return:
    """
    # front view input begins at conv1 and top view input begins at conv2
    # Stage 1
    front_conv1 = conv2d(_input=front_view_input, _conv_para=dvcnn_architecture['conv1'], name='conv1', reuse=False)
    front_bn1 = batch_norm(_input=front_conv1, name='bn1', reuse=False)
    front_relu1 = activate(_input=front_bn1, _activate_para=dvcnn_architecture['relu1'], name='relu1', reuse=False)
    front_pool1 = max_pool(_input=front_relu1, _max_pool_para=dvcnn_architecture['pool1'], name='pool1', reuse=False)

    # Stage 2
    front_conv2 = conv2d(_input=front_pool1, _conv_para=dvcnn_architecture['conv2_front'], name='conv2_front',
                         reuse=False)
    front_bn2 = batch_norm(_input=front_conv2, name='bn2_front', reuse=False)
    front_relu2 = activate(_input=front_bn2, _activate_para=dvcnn_architecture['relu2'], name='relu2', reuse=False)
    front_pool2 = max_pool(_input=front_relu2, _max_pool_para=dvcnn_architecture['pool2'], name='pool2', reuse=False)

    top_conv2 = conv2d(_input=top_view_input, _conv_para=dvcnn_architecture['conv2_top'], name='conv2_top', reuse=False)
    top_bn2 = batch_norm(_input=top_conv2, name='bn2_top', reuse=False)
    top_relu2 = activate(_input=top_bn2, _activate_para=dvcnn_architecture['relu2'], name='relu2', reuse=True)
    top_pool2 = max_pool(_input=top_relu2, _max_pool_para=dvcnn_architecture['pool2'], name='pool2', reuse=True)

    # Stage 3
    front_conv3 = conv2d(_input=front_pool2, _conv_para=dvcnn_architecture['conv3'], name='conv3', reuse=False)
    front_bn3 = batch_norm(_input=front_conv3, name='bn3', reuse=False)
    front_relu3 = activate(_input=front_bn3, _activate_para=dvcnn_architecture['relu3'], name='relu3', reuse=False)
    front_pool3 = max_pool(_input=front_relu3, _max_pool_para=dvcnn_architecture['pool3'], name='pool3', reuse=False)

    top_conv3 = conv2d(_input=top_pool2, _conv_para=dvcnn_architecture['conv3'], name='conv3', reuse=True)
    top_bn3 = batch_norm(_input=top_conv3, name='bn3', reuse=True)
    top_relu3 = activate(_input=top_bn3, _activate_para=dvcnn_architecture['relu3'], name='relu3', reuse=True)
    top_pool3 = max_pool(_input=top_relu3, _max_pool_para=dvcnn_architecture['pool3'], name='pool3', reuse=True)

    # Stage 4
    front_conv4 = conv2d(_input=front_pool3, _conv_para=dvcnn_architecture['conv4'], name='conv4', reuse=False)
    front_bn4 = batch_norm(_input=front_conv4, name='bn4', reuse=False)
    front_relu4 = activate(_input=front_bn4, _activate_para=dvcnn_architecture['relu4'], name='relu4', reuse=False)
    front_pool4 = max_pool(_input=front_relu4, _max_pool_para=dvcnn_architecture['pool4'], name='pool4', reuse=False)

    top_conv4 = conv2d(_input=top_pool3, _conv_para=dvcnn_architecture['conv4'], name='conv4', reuse=True)
    top_bn4 = batch_norm(_input=top_conv4, name='bn4', reuse=True)
    top_relu4 = activate(_input=top_bn4, _activate_para=dvcnn_architecture['relu4'], name='relu4', reuse=True)
    top_pool4 = max_pool(_input=top_relu4, _max_pool_para=dvcnn_architecture['pool4'], name='pool4', reuse=True)

    # Stage 5
    front_conv5 = conv2d(_input=front_pool4, _conv_para=dvcnn_architecture['conv5'], name='conv5', reuse=False)
    front_bn5 = batch_norm(_input=front_conv5, name='bn5', reuse=False)
    front_relu5 = activate(_input=front_bn5, _activate_para=dvcnn_architecture['relu5'], name='relu5', reuse=False)
    front_pool5 = max_pool(_input=front_relu5, _max_pool_para=dvcnn_architecture['pool5'], name='pool5', reuse=False)

    top_conv5 = conv2d(_input=top_pool4, _conv_para=dvcnn_architecture['conv5'], name='conv5', reuse=True)
    top_bn5 = batch_norm(_input=top_conv5, name='bn5', reuse=True)
    top_relu5 = activate(_input=top_bn5, _activate_para=dvcnn_architecture['relu5'], name='relu5', reuse=True)
    top_pool5 = max_pool(_input=top_relu5, _max_pool_para=dvcnn_architecture['pool5'], name='pool5', reuse=True)

    # Stage 6
    front_fc6 = fully_connect(_input=front_pool5, _fc_para=dvcnn_architecture['fc6'], name='fc6', reuse=False)
    front_bn6 = batch_norm(_input=front_fc6, name='bn6', reuse=False)
    front_relu6 = activate(_input=front_bn6, _activate_para=dvcnn_architecture['relu6'], name='relu6', reuse=False)

    top_fc6 = fully_connect(_input=top_pool5, _fc_para=dvcnn_architecture['fc6'], name='fc6', reuse=True)
    top_bn6 = batch_norm(_input=top_fc6, name='bn6', reuse=True)
    top_relu6 = activate(_input=top_bn6, _activate_para=dvcnn_architecture['relu6'], name='relu6', reuse=True)

    # Stage 7
    concat7 = concat(_input=[front_relu6, top_relu6], _concat_para=dvcnn_architecture['concat7'], name='concat7')

    # Stage 8
    fc8 = fully_connect(_input=concat7, _fc_para=dvcnn_architecture['fc8'], name='fc8', reuse=False)

    # Convert fc8 from matrix into a vector
    out_put = tf.reshape(tensor=fc8, shape=[-1, dvcnn_architecture['fc8']['ksize'][-1]])

    return out_put


def test_net(model_path, weights_path, lane_dir, non_lane_dir):
    """
    Test DVCNN network
    :param model_path: DVCNN json model path
    :param weights_path: DVCNN network weights file path(ckpt file)
    :param lane_dir: lane line sample dir where contains top_view and front_view folder
    :param non_lane_dir: non lane line sample dir where contains top_view and front_view folder
    :return:
    """
    # read dvcnn architecture parameter
    dvcnn_architecture = read_json_model(json_model_path=model_path)
    # provide the data for testing
    provider = data_provider.DataProvider(lane_dir=lane_dir, not_lane_dir=non_lane_dir)

    test_top_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='top_input')
    test_front_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='front_input')

    test_batch_data = provider.next_batch(batch_size=640)
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

    dvcnn_out = build_dvcnn(top_view_input=test_top_input_tensor,
                            front_view_input=test_front_input_tensor,
                            dvcnn_architecture=dvcnn_architecture)
    preds = tf.argmax(tf.nn.softmax(dvcnn_out), 1)

    lane_sample_nums = np.count_nonzero(test_label)
    non_lane_sample_nums = len(test_label) - lane_sample_nums

    saver = tf.train.Saver()

    sess = tf.Session()

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        prediction, fv_image_list = sess.run([preds, test_front_input_tensor],
                                             feed_dict={test_front_input_tensor: test_front_input,
                                                        test_top_input_tensor: test_top_input})
        diff = prediction - test_label
        correct_prediction = np.count_nonzero(diff == 0)
        accuracy = correct_prediction / len(test_label)
        for index, fv_image in enumerate(fv_image_list):
            print('Image file is {:s} and label is {:d}'.format(test_front_filename[index], prediction[index]))
            plt.imshow(np.uint8(fv_image[:, :, (2, 1, 0)]))
            plt.show()
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
    # read dvcnn architecture parameter
    dvcnn_architecture = read_json_model(json_model_path=model_path)
    # provide the data for testing
    provider = data_provider.DataProvider(lane_dir=lane_dir, not_lane_dir=non_lane_dir)

    test_top_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='top_input')
    test_front_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='front_input')

    dvcnn_out = build_dvcnn(top_view_input=test_top_input_tensor,
                            front_view_input=test_front_input_tensor,
                            dvcnn_architecture=dvcnn_architecture)
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
    print('Done')
    return


if __name__ == '__main__':
    test_net_lane(model_path='DVCNN/model_def/DVCNN.json', weights_path='DVCNN/model/dvcnn.ckpt-1199',
                  lane_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE_TEST/lane_line',
                  non_lane_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE_TEST/non_lane_line')
    select_lane_fv_sample(lane_result_file='DVCNN/lane_result.txt')
