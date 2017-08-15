import numpy as np
import cv2
import os
import sys
try:
    from cv2 import cv2
except ImportError:
    pass

front_view_ctrl_point = [(98, 701), (770, 701), (291, 541), (645, 541)]
top_view_ctrl_point = [(425, 701), (525, 701), (425, 600), (525, 600)]


def inverse_perspective_map(image):
    """
    Convert front view image to top view image
    :param image:
    :return:
    """
    fv_ctrl_point = np.array(front_view_ctrl_point).astype(dtype=np.float32)
    top_ctrl_point = np.array(top_view_ctrl_point).astype(dtype=np.float32)

    warp_transform = cv2.getPerspectiveTransform(src=fv_ctrl_point, dst=top_ctrl_point)
    warped_image = cv2.warpPerspective(src=image, M=warp_transform, dsize=(1000, 700))
    return warped_image


def perspective_map(image):
    """
    Convert top view image to front view image
    :param image:
    :return:
    """
    top_ctrl_point = np.array(top_view_ctrl_point).astype(dtype=np.float32)
    fv_ctrl_point = np.array(front_view_ctrl_point).astype(dtype=np.float32)

    warp_transform = cv2.getPerspectiveTransform(src=top_ctrl_point, dst=fv_ctrl_point)
    warped_image = cv2.warpPerspective(src=image, M=warp_transform, dsize=(1000, 700))
    return warped_image


def inverse_perspective_point(pt1):
    """
    map point in front view image into top view image
    :param pt1:
    :return:
    """
    fv_ctrl_point = np.array(front_view_ctrl_point).astype(dtype=np.float32)
    top_ctrl_point = np.array(top_view_ctrl_point).astype(dtype=np.float32)

    warp_transform = cv2.getPerspectiveTransform(src=fv_ctrl_point, dst=top_ctrl_point)
    pt_warp = cv2.perspectiveTransform(src=pt1, m=warp_transform)
    return pt_warp[0, 0, :]


def perspective_point(pt1):
    """
    map point in top view image into front view image
    :param pt1:
    :return:
    """
    pt1 = np.array([[pt1]], dtype=np.float32)

    top_ctrl_point = np.array(top_view_ctrl_point).astype(dtype=np.float32)
    fv_ctrl_point = np.array(front_view_ctrl_point).astype(dtype=np.float32)

    warp_transform = cv2.getPerspectiveTransform(src=top_ctrl_point, dst=fv_ctrl_point)
    pt_warp = cv2.perspectiveTransform(src=pt1, m=warp_transform)
    return pt_warp[0, 0, :]


def inverse():
    fv_image_src = '/home/baidu/DataBase/Road_Center_Line_DataBase/Origin/front_view'
    top_image_path = '/home/baidu/DataBase/Road_Center_Line_DataBase/Origin/top_view'

    if not os.path.exists(top_image_path):
        os.makedirs(top_image_path)

    for parents, _, filenames in os.walk(fv_image_src):
        for index, filename in enumerate(filenames):
            fv_img_id = os.path.join(parents, filename)
            fv_img = cv2.imread(fv_img_id, cv2.IMREAD_UNCHANGED)
            top_image = inverse_perspective_map(image=fv_img)
            top_image_save_path = os.path.join(top_image_path, filename.replace('fv', 'top'))
            cv2.imwrite(top_image_save_path, top_image)
            sys.stdout.write('\r>>Map {:d}/{:d} {:s}'.format(index+1, len(filenames),
                                                             os.path.split(top_image_save_path)[1]))
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    return

if __name__ == '__main__':
    inverse()
