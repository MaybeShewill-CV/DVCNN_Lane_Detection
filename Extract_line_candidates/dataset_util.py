"""
Some functions used to process the dataset
"""
import os


class Dataset(object):
    def __init__(self, dir_path, file_flag):
        if not os.path.exists(dir_path):
            raise ValueError('File {} doesn\'t exist'.format(dir_path))
        if type(file_flag) != str:
            raise TypeError('file_flag should be the file type e.g \'.jpg\'')
        self.__nums = 0
        self.__filelist = []
        self.__get_files(dir_path=dir_path, file_flag=file_flag)
        return

    def __get_files(self, dir_path, file_flag):
        for parents, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                if filename.endswith(file_flag):
                    path = os.path.join(parents, filename)
                    # abspath = os.path.abspath(path)
                    self.__filelist.append(path)
                    self.__nums += 1
        return

    def get_filelist(self):
        return self.__filelist

    def get_filenums(self):
        return self.__nums
