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
from pprint import pprint

from DVCNN import preprocess
from DVCNN.data_provider import DataProvider
from DVCNN import dvcnn_model
from DVCNN.cnn_util import read_json_model
from Global_Configuration.config import cfg


def train_dvcnn(lane_dir, non_lane_dir):
    """
    Train DVCNN model
    :param lane_dir: where you store the lane line positive samples which should include folders front_view and top_view
    :param non_lane_dir: where you store the non lane line negative samples which should include folders
    front_view and top_view
    :return:
    """
    # Set train data provider
    provider = DataProvider(lane_dir=lane_dir, not_lane_dir=non_lane_dir)
    print(provider)

    # Set validation data provider
    val_provider = DataProvider(lane_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                         'DVCNN_SAMPLE/Validation/lane_line',
                                not_lane_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                             'DVCNN_SAMPLE/Validation/non_lane_line')
    print(val_provider)
    val_top_input = []
    val_front_input = []
    val_label_input = []
    val_batch_data = val_provider.next_batch(batch_size=cfg.TRAIN.VAL_BATCH_SIZE)

    for index, val_data in enumerate(val_batch_data):
        top_file_name = val_data[0]
        front_file_name = val_data[1]
        label = val_data[2]
        top_image = cv2.imread(top_file_name, cv2.IMREAD_UNCHANGED)
        top_image = cv2.resize(src=top_image, dsize=(64, 64))
        front_image = cv2.imread(front_file_name, cv2.IMREAD_UNCHANGED)
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

    # Set the dvcnn architecture
    dvcnn_architecture = read_json_model('DVCNN/model_def/DVCNN.json')

    # Set global training parameters
    training_epochs = cfg.TRAIN.EPOCHS
    display_step = cfg.TRAIN.DISPLAY_STEP
    test_display_step = cfg.TRAIN.TEST_DISPLAY_STEP

    # Set input tensors and augmentation processor
    train_top_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='top_input')
    train_front_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='front_input')

    test_top_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='top_input')
    test_front_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='front_input')

    augment_dict_whiten = {
        'minmax_normalization': False,
        'flip_horizon': False,
        'flip_vertical': False,
        'random_crop': {
            'need_random_crop': False
        },
        'random_brightness': {
            'need_random_brightness': False
        },
        'random_contrast': {
            'need_random_contrast': False
        },
        'std_normalization': False,
        'centralization': {
            'need_centralization': True,
            'mean_value': [103.939, 116.779, 123.68]
        }
    }

    train_top_input_tensor_aug = preprocessor.augment_image(self=preprocessor,
                                                            image=train_top_input_tensor,
                                                            augment_para_dict=augment_dict_whiten)
    train_front_input_tensor_aug = preprocessor.augment_image(self=preprocessor,
                                                              image=train_front_input_tensor,
                                                              augment_para_dict=augment_dict_whiten)
    train_top_input_tensor = tf.concat([train_top_input_tensor, train_top_input_tensor_aug], axis=0)
    train_front_input_tensor = tf.concat([train_front_input_tensor, train_front_input_tensor_aug], axis=0)
    train_label_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='train_label_input')

    test_top_input_tensor_aug = preprocessor.augment_image(self=preprocessor,
                                                           image=test_top_input_tensor,
                                                           augment_para_dict=augment_dict_whiten)
    test_front_input_tensor_aug = preprocessor.augment_image(self=preprocessor,
                                                             image=test_front_input_tensor,
                                                             augment_para_dict=augment_dict_whiten)
    test_top_input_tensor = tf.concat([test_top_input_tensor, test_top_input_tensor_aug], axis=0)
    test_front_input_tensor = tf.concat([test_front_input_tensor, test_front_input_tensor_aug], axis=0)
    test_label_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='test_label_input')

    # Set dvcnn model output tensor
    dvcnn_train_out = dvcnn_model.build_dvcnn(top_view_input=train_top_input_tensor,
                                              front_view_input=train_front_input_tensor,
                                              dvcnn_architecture=dvcnn_architecture)

    dvcnn_test_out = dvcnn_model.build_dvcnn_test(top_view_input=test_top_input_tensor,
                                                  front_view_input=test_front_input_tensor,
                                                  dvcnn_architecture=dvcnn_architecture)
    correct_preds_train = tf.equal(tf.argmax(tf.nn.softmax(dvcnn_train_out), 1), tf.argmax(train_label_input_tensor, 1))
    accuracy_train = tf.reduce_mean(tf.cast(correct_preds_train, tf.float32), name='accuracy_train')
    correct_preds_val = tf.equal(tf.argmax(tf.nn.softmax(dvcnn_test_out), 1), tf.argmax(test_label_input_tensor, 1))
    accuracy_test = tf.reduce_mean(tf.cast(correct_preds_val, tf.float32), name='accuracy_val')

    # Set loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_label_input_tensor,
                                                                                logits=dvcnn_train_out, name='cost'))
    l2_loss = 0.0
    for v in tf.trainable_variables():
        if not v.name[:-2].endswith('bias'):
            l2_loss += tf.nn.l2_loss(t=v, name='{}_l2_loss'.format(v.name[:-2]))

    total_cost = cross_entropy_loss + l2_loss * cfg.TRAIN.L2_DECAY_RATE

    # Set the global optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=cfg.TRAIN.LEARNING_RATE, global_step=global_step,
                                               decay_steps=cfg.TRAIN.LR_DECAY_STEPS, decay_rate=cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(momentum=cfg.TRAIN.MOMENTUM,
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
    tf.summary.scalar(name='Test Accuracy', tensor=accuracy_test)
    tf.summary.scalar(name='Learning Rate', tensor=learning_rate)
    mergen_summary_op = tf.summary.merge_all()

    # Set saver configuration
    saver = tf.train.Saver()
    save_path = 'DVCNN/model/dvcnn.ckpt'

    # Set sess configuration
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = cfg.TRAIN.GPU_MEMORY_FRACTION
    config.gpu_options.allow_growth = cfg.TRAIN.TF_ALLOW_GROWTH

    # Set tensorflow session
    sess = tf.Session(config=config)

    print('DVCNN training parameters are as follows:')
    pprint(cfg)

    with sess.as_default():

        init = tf.global_variables_initializer()
        sess.run(init)

        summary_writer = tf.summary.FileWriter(tboard_save_path)
        summary_writer.add_graph(sess.graph)

        for epoch in range(training_epochs):
            train_top_input = []
            train_front_input = []
            train_label_input = []
            train_batch_data = provider.next_batch(batch_size=cfg.TRAIN.BATCH_SIZE)
            for j in range(len(train_batch_data)):
                top_file_name = train_batch_data[j][0]
                front_file_name = train_batch_data[j][1]
                label = train_batch_data[j][2]
                top_image = cv2.imread(top_file_name, cv2.IMREAD_UNCHANGED)
                top_image = cv2.resize(src=top_image, dsize=(64, 64))
                front_image = cv2.imread(front_file_name, cv2.IMREAD_UNCHANGED)
                front_image = cv2.resize(src=front_image, dsize=(128, 128))
                train_top_input.append(top_image)
                train_front_input.append(front_image)
                train_label_input.append(label)

            for kk in range(len(train_label_input)):
                if train_label_input[kk] == 1:
                    train_label_input[kk] = [0, 1]
                else:
                    train_label_input[kk] = [1, 0]

            _, c, train_out, test_out, train_accuracy, test_accuracy, summary = sess.run(
                [optimizer, total_cost, dvcnn_train_out, dvcnn_test_out, accuracy_train, accuracy_test,
                 mergen_summary_op],
                feed_dict={train_top_input_tensor: train_top_input,
                           train_front_input_tensor: train_front_input,
                           train_label_input_tensor: train_label_input,
                           test_top_input_tensor: val_top_input,
                           test_front_input_tensor: val_front_input,
                           test_label_input_tensor: val_label_input})

            summary_writer.add_summary(summary=summary, global_step=epoch)

            if epoch % display_step == 0:
                print('Epoch: {:04d} cost= {:9f} accuracy= {:9f}'.format(epoch + 1, c, train_accuracy))

            if epoch % test_display_step == 0:
                print('Epoch: {:04d} test_accuracy= {:9f}'.format(epoch + 1, test_accuracy))

            saver.save(sess=sess, save_path=save_path, global_step=epoch)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane_line_dir', type=str, help='Where you store the positive lane line samples')
    parser.add_argument('--non_lane_line_dir', type=str, help='Where you store the negative lane line samples')

    args = parser.parse_args()

    train_dvcnn(lane_dir=args.lane_line_dir, non_lane_dir=args.non_lane_line_dir)
