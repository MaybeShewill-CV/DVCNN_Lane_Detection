"""
Use weights hat-like filter to filter the top view image and extract the roi patch and its' corresponding roi patch in
front view image and save them into data lane_line which are supposed to be selected by hands later. This function is
mainly used during training time to support the sample selection work
"""
import os

import cv2
import numpy as np
import sys
try:
    from cv2 import cv2
except ImportError:
    pass

import Extract_line_candidates.extract_candidate
import inverse_perspective_map

START_X = 325
START_Y = 327
CROP_WIDTH = 325
CROP_HEIGHT = 325


def __get_rect(rotbox):
    """
    Get the min surrounding box around the rotate box
    :param rotbox: rotate box
    :return:
    """
    x_list = [rotbox[0][0], rotbox[1][0], rotbox[2][0], rotbox[3][0]]
    y_list = [rotbox[0][1], rotbox[1][1], rotbox[2][1], rotbox[3][1]]

    xmin = np.amin(a=np.array(x_list), axis=0)
    xmax = np.amax(a=np.array(x_list), axis=0)
    ymin = np.amin(a=np.array(y_list), axis=0)
    ymax = np.amax(a=np.array(y_list), axis=0)

    rect = np.array([xmin, ymin, xmax, ymax])
    return rect


def extract_and_save_roi_patch(top_image_dir, fv_image_dir, top_view_roi_save_path, front_view_roi_save_path):
    """
    extract roi patch from top view image and save the roi patch and its' corresponding front view roi patch
    :param top_image_dir: the path where you store the top view image
    :param fv_image_dir: the path where you store the front view image
    :param top_view_roi_save_path: the path where you store the top view roi patch
    :param front_view_roi_save_path: the path where you store the front view roi patch
    :return:
    """
    if not os.path.exists(top_image_dir):
        raise ValueError('{:s} doesn\'t exist'.format(top_image_dir))
    if not os.path.exists(fv_image_dir):
        raise ValueError('{:s} doesn\'t exist'.format(fv_image_dir))
    if not os.path.exists(top_view_roi_save_path):
        os.makedirs(top_view_roi_save_path)
    if not os.path.exists(front_view_roi_save_path):
        os.makedirs(front_view_roi_save_path)

    res_info = Extract_line_candidates.extract_candidate.extract_all(top_image_dir)

    extract_count = 0
    for image_id, info in res_info.items():
        # read top view image
        top_image_id = image_id
        top_image = cv2.imread(os.path.join(top_image_dir, top_image_id), cv2.IMREAD_UNCHANGED)
        # read front view image
        fv_image_id = image_id.replace('top', 'fv')
        fv_image = cv2.imread(os.path.join(fv_image_dir, fv_image_id), cv2.IMREAD_UNCHANGED)

        # get roi information from top view image
        rrect_list = info['rrect']

        for index, rrect in enumerate(rrect_list):
            rotbox = np.int0(cv2.boxPoints(rrect))
            # get the min rect surrounding the rotate rect
            top_minrect = __get_rect(rotbox=rotbox)
            # get the top roi
            top_roi = top_image[top_minrect[1]:top_minrect[3], top_minrect[0]:top_minrect[2], :]
            # top view roi save name constructed as xxxx_top_index_ltx_lty_rbx_rby.jpg to reserve the position
            # information in the top view image
            top_roi_save_id = '{:s}_{:d}.jpg'.format(top_image_id[:-4], index)
            top_roi_save_id = os.path.join(top_view_roi_save_path, top_roi_save_id)
            cv2.imwrite(top_roi_save_id, top_roi)

            for j in range(4):
                # remap the rotate rect position from the cropped top view image to the origin top view image which is
                # correspond to the front view image. The cropped top view image is cropped from the origin top view
                # image at position [START_X, START_Y] with [CROP_WIDTH, CROP_HEIGHT] width and height
                rotbox[j] = np.add(rotbox[j], [START_X, START_Y])
                # remap the position from top view image to front view image
                rotbox[j] = inverse_perspective_map.perspective_point(rotbox[j])
            # get the min surrounding rect of the rotate rect
            fv_minrect = __get_rect(rotbox)
            fv_roi = fv_image[fv_minrect[1]:fv_minrect[3], fv_minrect[0]:fv_minrect[2], :]
            # fv roi image id was constructed in the same way as top roi image id
            fv_roi_save_id = '{:s}_{:d}.jpg'.format(fv_image_id[:-4], index)
            fv_roi_save_id = os.path.join(front_view_roi_save_path, fv_roi_save_id)
            cv2.imwrite(fv_roi_save_id, fv_roi)
        extract_count += 1
        sys.stdout.write('\r>>Extract {:d}/{:d} {:s}'.format(extract_count, len(res_info), image_id))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    return


if __name__ == '__main__':
    extract_and_save_roi_patch(top_image_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/Origin/top_view_crop',
                               fv_image_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/Origin/fv_view',
                               top_view_roi_save_path='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                                      'Extract_Roi/top_view',
                               front_view_roi_save_path='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                                        'Extract_Roi/front_view')
    print('Done')
