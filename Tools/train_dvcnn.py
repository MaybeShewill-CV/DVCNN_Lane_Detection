#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : train_dvcnn.py
"""
Train the DVCNN model
"""
import os
import cv2
import argparse
import tensorflow as tf
try:
    from cv2 import cv2
except ImportError:
    pass
import pprint

from DVCNN import data_provider, dvcnn_model, preprocess
from DVCNN.model_def import dvcnn_global_variable
from Global_Configuration import config


def train_dvcnn(lane_dir, non_lane_dir, json_model_path):
    """
    Train DVCNN model
    :param lane_dir: where you store the lane line positive samples which should include folders front_view and top_view
    :param non_lane_dir: where you store the non lane line negative samples which should include folders
    :param json_model_path: where you store the json model file
    front_view and top_view
    :return:
    """
    # Set train data provider
    provider = data_provider.DataProvider(lane_dir=lane_dir, not_lane_dir=non_lane_dir)
    print(provider)

    # Set validation data provider
    val_provider = data_provider.DataProvider(lane_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                              'DVCNN_SAMPLE/Validation/lane_line',
                                              not_lane_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                              'DVCNN_SAMPLE/Validation/non_lane_line')
    print(val_provider)
    val_top_input = []
    val_front_input = []
    val_label_input = []
    val_batch_data = val_provider.next_batch(batch_size=config.cfg.TRAIN.VAL_BATCH_SIZE)

    for index, val_data in enumerate(val_batch_data):
        top_file_name = val_data[0]
        front_file_name = val_data[1]
        label = val_data[2]
        top_image = cv2.imread(top_file_name, cv2.IMREAD_UNCHANGED)[:, :, (2, 1, 0)]
        top_image = cv2.resize(src=top_image, dsize=(64, 64))
        front_image = cv2.imread(front_file_name, cv2.IMREAD_UNCHANGED)[:, :, (2, 1, 0)]
        front_image = cv2.resize(src=front_image, dsize=(128, 128))
        val_top_input.append(top_image)
        val_front_input.append(front_image)
        val_label_input.append(label)

    for kk in range(len(val_label_input)):
        if val_label_input[kk] == 1:
            val_label_input[kk] = [0, 1]
        else:
            val_label_input[kk] = [1, 0]

    # Set train image preprocessor
    preprocessor = preprocess.Preprocessor()

    # Set global training parameters
    training_epochs = config.cfg.TRAIN.EPOCHS
    display_step = config.cfg.TRAIN.DISPLAY_STEP
    val_display_step = config.cfg.TRAIN.TEST_DISPLAY_STEP

    # Set input tensors and augmentation processor
    train_top_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='top_input')
    train_front_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='front_input')
    train_label_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='train_label_input')
    train_label_input_tensor_concat = train_label_input_tensor
    train_top_input_tensor_concat = train_top_input_tensor
    train_front_input_tensor_concat = train_front_input_tensor

    val_top_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='top_input')
    val_front_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='front_input')
    val_label_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='test_label_input')

    if config.cfg.TRAIN.USE_HORIZON_FLIP:
        top_tmp = preprocessor.augment_image(image=train_top_input_tensor, function_flag='flip_horizon',
                                             function_params=dict())
        front_tmp = preprocessor.augment_image(image=train_front_input_tensor, function_flag='flip_horizon',
                                               function_params=dict())
        train_top_input_tensor_concat = tf.concat([train_top_input_tensor_concat, top_tmp], axis=0)
        train_front_input_tensor_concat = tf.concat([train_front_input_tensor_concat, front_tmp], axis=0)
        train_label_input_tensor_concat = tf.concat([train_label_input_tensor_concat, train_label_input_tensor], axis=0)

    if config.cfg.TRAIN.USE_VERTICAL_FLIP:
        top_tmp = preprocessor.augment_image(image=train_top_input_tensor, function_flag='flip_vertical',
                                             function_params=dict())
        front_tmp = preprocessor.augment_image(image=train_front_input_tensor, function_flag='flip_vertical',
                                               function_params=dict())
        train_top_input_tensor_concat = tf.concat([train_top_input_tensor_concat, top_tmp], axis=0)
        train_front_input_tensor_concat = tf.concat([train_front_input_tensor_concat, front_tmp], axis=0)
        train_label_input_tensor_concat = tf.concat([train_label_input_tensor_concat, train_label_input_tensor], axis=0)

    if config.cfg.TRAIN.USE_RANDOM_CONTRAST:
        top_tmp = preprocessor.augment_image(image=train_top_input_tensor, function_flag='random_contrast',
                                             function_params=dvcnn_global_variable.DVCNN_AUGMENTATION_DICTS
                                             ['random_contrast'])
        front_tmp = preprocessor.augment_image(image=train_front_input_tensor, function_flag='random_contrast',
                                               function_params=dvcnn_global_variable.DVCNN_AUGMENTATION_DICTS
                                               ['random_contrast'])
        train_top_input_tensor_concat = tf.concat([train_top_input_tensor_concat, top_tmp], axis=0)
        train_front_input_tensor_concat = tf.concat([train_front_input_tensor_concat, front_tmp], axis=0)
        train_label_input_tensor_concat = tf.concat([train_label_input_tensor_concat, train_label_input_tensor], axis=0)

    if config.cfg.TRAIN.USE_RANDOM_BRIGHTNESS:
        top_tmp = preprocessor.augment_image(image=train_top_input_tensor, function_flag='random_brightness',
                                             function_params=dvcnn_global_variable.DVCNN_AUGMENTATION_DICTS
                                             ['random_brightness'])
        front_tmp = preprocessor.augment_image(image=train_front_input_tensor, function_flag='random_brightness',
                                               function_params=dvcnn_global_variable.DVCNN_AUGMENTATION_DICTS
                                               ['random_brightness'])
        train_top_input_tensor_concat = tf.concat([train_top_input_tensor_concat, top_tmp], axis=0)
        train_front_input_tensor_concat = tf.concat([train_front_input_tensor_concat, front_tmp], axis=0)
        train_label_input_tensor_concat = tf.concat([train_label_input_tensor_concat, train_label_input_tensor], axis=0)

    # Set dvcnn model output tensor
    dvcnn = dvcnn_model.DVCNNBuilder(json_model_path=json_model_path)

    dvcnn_train_out = dvcnn.build_dvcnn(top_view_input=train_top_input_tensor_concat,
                                        front_view_input=train_front_input_tensor_concat)

    dvcnn_val_out = dvcnn.build_dvcnn_val(top_view_input=val_top_input_tensor,
                                          front_view_input=val_front_input_tensor)

    correct_preds_train = tf.equal(tf.argmax(tf.nn.softmax(dvcnn_train_out), 1),
                                   tf.argmax(train_label_input_tensor_concat, 1))
    accuracy_train = tf.reduce_mean(tf.cast(correct_preds_train, tf.float32), name='accuracy_train')
    correct_preds_val = tf.equal(tf.argmax(tf.nn.softmax(dvcnn_val_out), 1), tf.argmax(val_label_input_tensor, 1))
    accuracy_val = tf.reduce_mean(tf.cast(correct_preds_val, tf.float32), name='accuracy_val')

    # Set loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_label_input_tensor_concat,
                                                                                logits=dvcnn_train_out, name='cost'))
    l2_loss = 0.0
    for v in tf.trainable_variables():
        if not v.name[:-2].endswith('bias'):
            l2_loss += tf.nn.l2_loss(t=v, name='{}_l2_loss'.format(v.name[:-2]))

    total_cost = cross_entropy_loss + l2_loss * config.cfg.TRAIN.L2_DECAY_RATE

    # Set the global optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=config.cfg.TRAIN.LEARNING_RATE,
                                               global_step=global_step,
                                               decay_steps=config.cfg.TRAIN.LR_DECAY_STEPS,
                                               decay_rate=config.cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(momentum=config.cfg.TRAIN.MOMENTUM,
                                               learning_rate=learning_rate).minimize(loss=total_cost,
                                                                                     global_step=global_step)

    # Set tf summary
    tboard_save_path = 'DVCNN/tboard'
    if not os.path.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    tf.summary.scalar(name='Cross entropy loss', tensor=cross_entropy_loss)
    tf.summary.scalar(name='L2 loss', tensor=l2_loss)
    tf.summary.scalar(name='Total loss', tensor=total_cost)
    tf.summary.scalar(name='Train Accuracy', tensor=accuracy_train)
    tf.summary.scalar(name='Test Accuracy', tensor=accuracy_val)
    tf.summary.scalar(name='Learning Rate', tensor=learning_rate)
    mergen_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    save_path = 'DVCNN/model/dvcnn.ckpt'

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # Set tensorflow session
    sess = tf.Session(config=sess_config)

    print('DVCNN training parameters are as follows:')
    pprint.pprint(config.cfg)

    with sess.as_default():

        init = tf.global_variables_initializer()
        sess.run(init)

        summary_writer = tf.summary.FileWriter(tboard_save_path)
        summary_writer.add_graph(sess.graph)

        for epoch in range(training_epochs):
            train_top_input = []
            train_front_input = []
            train_label_input = []
            train_batch_data = provider.next_batch(batch_size=config.cfg.TRAIN.BATCH_SIZE)
            for j, train_data in enumerate(train_batch_data):
                top_file_name = train_data[0]
                front_file_name = train_data[1]
                label = train_data[2]
                top_image = cv2.imread(top_file_name, cv2.IMREAD_UNCHANGED)[:, :, (2, 1, 0)]
                top_image = cv2.resize(src=top_image, dsize=(64, 64))
                front_image = cv2.imread(front_file_name, cv2.IMREAD_UNCHANGED)[:, :, (2, 1, 0)]
                front_image = cv2.resize(src=front_image, dsize=(128, 128))
                train_top_input.append(top_image)
                train_front_input.append(front_image)
                train_label_input.append(label)

            for kk in range(len(train_label_input)):
                if train_label_input[kk] == 1:
                    train_label_input[kk] = [0, 1]
                else:
                    train_label_input[kk] = [1, 0]

            _, c, train_out, val_out, train_accuracy, val_accuracy, summary = sess.run(
                [optimizer, total_cost, dvcnn_train_out, dvcnn_val_out, accuracy_train, accuracy_val,
                 mergen_summary_op],
                feed_dict={train_top_input_tensor: train_top_input,
                           train_front_input_tensor: train_front_input,
                           train_label_input_tensor: train_label_input,
                           val_top_input_tensor: val_top_input,
                           val_front_input_tensor: val_front_input,
                           val_label_input_tensor: val_label_input})

            summary_writer.add_summary(summary=summary, global_step=epoch)

            if epoch % display_step == 0:
                print('Epoch: {:04d} cost= {:9f} accuracy= {:9f}'.format(epoch + 1, c, train_accuracy))

            if epoch % val_display_step == 0:
                print('Epoch: {:04d} test_accuracy= {:9f}'.format(epoch + 1, val_accuracy))

            saver.save(sess=sess, save_path=save_path, global_step=epoch)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane_line_dir', type=str, help='Where you store the positive lane line samples')
    parser.add_argument('--non_lane_line_dir', type=str, help='Where you store the negative lane line samples')
    parser.add_argument('--model_path', type=str, help='Where you store the json model file')

    args = parser.parse_args()

    train_dvcnn(lane_dir=args.lane_line_dir, non_lane_dir=args.non_lane_line_dir, json_model_path=args.model_path)
