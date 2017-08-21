#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : utils.py
"""
Define some global used util functions
"""


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