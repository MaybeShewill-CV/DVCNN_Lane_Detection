#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : cnn_util.py
"""
Some dvcnn util functions
"""
import tensorflow as tf
import os
import json
import collections

from DVCNN.model_def import dvcnn_global_variable


def write_dvcnn_model(json_path):
    """
    Write model definition into json file
    :param json_path: path to store model
    :return:
    """
    if _model_json_exist(json_path):
        print('{} already exist'.format(json_path))
        return

    dvcnn_model = _convert_to_ordered_dict(model_dict=dvcnn_global_variable.DVCNN_ARCHITECTURE)
    jsonobj = json.dumps(dvcnn_model, indent=4)
    file = open(json_path, 'w')
    file.write(jsonobj)
    file.close()


def _model_json_exist(json_path):
    """
    Check if the json file exists
    :param json_path:
    :return:
    """
    return os.path.isfile(json_path)


def _convert_to_ordered_dict(model_dict):
    """
    Convert model dict into orderdict
    :param model_dict: model dict
    :return:
    """
    order_dict = collections.OrderedDict()
    for layer_name, layer_para in model_dict.items():
        order_dict[layer_name] = layer_para
    return order_dict


def read_json_model(json_model_path):
    """
    Read json model file into orderddict
    :param json_model_path:
    :return:
    """
    if not os.path.exists(json_model_path):
        raise ValueError('{:s} doesn\'t exist'.format(json_model_path))
    if not json_model_path.endswith('.json'):
        raise ValueError('model file should be a json file')

    with open(json_model_path, 'r') as f:
        jsonobj = json.load(f, object_pairs_hook=collections.OrderedDict)
        f.close()
    return jsonobj
