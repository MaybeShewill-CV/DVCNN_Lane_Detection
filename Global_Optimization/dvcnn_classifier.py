"""
Use the DVCNN model to classify if the roi is a lane line roi or not
"""
import os.path as ops
import tensorflow as tf

from DVCNN.dvcnn_model import build_dvcnn, build_dvcnn_test
from DVCNN.cnn_util import read_json_model


class DVCNNClassifier(object):
    def __init__(self, model_file, weights_file):
        """
        Initialize the dvcnn model
        :param model_file: the dvcnn architecture definition file path
        :param weights_file: the weights file path
        """
        if not ops.isfile(model_file):
            raise ValueError('{:s} is not a valid file'.format(model_file))

        self.__model_file = model_file
        self.__weights_file = weights_file
        self.__dvcnn_architecture = read_json_model(json_model_path=model_file)
        return

    def predict(self, top_view_image_list, front_view_image_list):
        """
        Use the top view and front view image pair to predict if the image is a lane line
        :param top_view_image_list:
        :param front_view_image_list:
        :return: prediction 1: lane line 0: non lane line
        """
        if top_view_image_list is None or front_view_image_list is None:
            raise ValueError('Image data is invalid')

        top_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='top_input')
        fv_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name='fv_input')

        if len(tf.trainable_variables()) == 0:
            dvcnn_out = build_dvcnn(top_view_input=top_input_tensor,
                                    front_view_input=fv_input_tensor,
                                    dvcnn_architecture=self.__dvcnn_architecture)
        else:
            dvcnn_out = build_dvcnn_test(top_view_input=top_input_tensor,
                                         front_view_input=fv_input_tensor,
                                         dvcnn_architecture=self.__dvcnn_architecture)

        preds = tf.argmax(tf.nn.softmax(dvcnn_out), 1)
        scores_op = tf.nn.softmax(logits=dvcnn_out)

        # set tf sess config
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = False

        saver = tf.train.Saver()

        sess = tf.Session(config=config)

        with sess.as_default():
            # restore the weights
            saver.restore(sess=sess, save_path=self.__weights_file)

            predictions, scores = sess.run([preds, scores_op], feed_dict={top_input_tensor: top_view_image_list,
                                                                          fv_input_tensor: front_view_image_list})
        result = []
        for index, prediction in enumerate(predictions):
            result.append((prediction, scores[index, prediction]))
        return result
