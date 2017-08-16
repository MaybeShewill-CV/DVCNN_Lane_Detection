import cv2
import os.path as ops
import os
import sys
import shutil
import time
try:
    from cv2 import cv2
except ImportError:
    pass

sys.path.append(os.getcwd())


def select_sample_from_top_view(fv_view_dir, src_top_view_dir, res_save_dir):
    """
    Select sample from src top view dir which is listed in fv view dir
    :param fv_view_dir:
    :param src_top_view_dir:
    :param res_save_dir:
    :return:
    """
    if not ops.exists(fv_view_dir) or not ops.exists(src_top_view_dir):
        raise ValueError('{:s} or {:s} doesn\'t exist'.format(fv_view_dir, src_top_view_dir))
    if not ops.exists(res_save_dir):
        os.makedirs(res_save_dir)

    for parents, _, filenames in os.walk(fv_view_dir):
        for index, filename in enumerate(filenames):
            fv_file_name = ops.join(fv_view_dir, filename)
            top_file_name = ops.join(src_top_view_dir, filename.replace('fv', 'top'))
            dst_file_path = ops.join(res_save_dir, filename.replace('fv', 'top'))
            fv_image = cv2.imread(fv_file_name)
            top_image = cv2.imread(top_file_name)

            if fv_image is None or top_image is None:
                print('{:s} image pair is invalid'.format(filename.find('.')))
                continue
            shutil.copyfile(top_file_name, dst_file_path)
            sys.stdout.write('\r>>Selecting {:d}/{:d} {:s}'.format(index+1, len(filenames),
                                                                   filename.replace('fv', 'top')))
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    return


if __name__ == '__main__':
    t_start = time.time()
    select_sample_from_top_view(fv_view_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                            'DVCNN_SAMPLE_TEST/lane_line/front_view_lane_line_for_training',
                                src_top_view_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                                 'DVCNN_SAMPLE_TEST/lane_line/top_view',
                                res_save_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                             'DVCNN_SAMPLE_TEST/lane_line/top_view_lane_line_for_training')
    select_sample_from_top_view(fv_view_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                            'DVCNN_SAMPLE_TEST/lane_line/front_view_non_lane_line_for_training',
                                src_top_view_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                                 'DVCNN_SAMPLE_TEST/lane_line/top_view',
                                res_save_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                             'DVCNN_SAMPLE_TEST/lane_line/top_view_non_lane_line_for_training')
    print('Elapsed time {:4f}s'.format(time.time() - t_start))
