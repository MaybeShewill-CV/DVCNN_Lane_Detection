"""
Binarized the weight hat-like filter result image via intensity normalizing and thresholding.
"""
import math

import cv2
import numpy as np

try:
    from cv2 import cv2
except ImportError:
    pass

from Global_Configuration.imdb import Roidb
from Extract_line_candidates.inverse_perspective_map import perspective_point
from Global_Configuration.config import cfg

START_X = cfg.ROI.TOP_CROP_START_X
START_Y = cfg.ROI.TOP_CROP_START_Y
(WARPED_IMAGE_WIDTH, WARPED_IMAGE_HEIGHT) = cfg.ROI.WARPED_SIZE


def __calculate_line_degree(pt1, pt2):
    """
    Calculate the line degree(the angle between the line and the x axis)
    :param pt1: start point of the line
    :param pt2: end point of the line
    :return: the degree of the line
    """
    if(pt1[0] - pt2[0]) != 0:
        curlineangle = math.atan((pt2[1] - pt1[1]) / (pt2[0] - pt1[0]))
        if curlineangle < 0:
            curlineangle += math.pi
    else:
        curlineangle = math.pi / 2.0

    return curlineangle*180.0 / math.pi


def __get_rrect_degree(_rrect):
    """
    Calculate the rotate degree of the rotate rect(angle between the longer side of the rotate rect and the x axis)
    :param _rrect: Rotate degree
    :return:
    """
    points = cv2.boxPoints(box=_rrect)
    firstline_length = math.pow((points[1][0] - points[0][0]), 2) + math.pow((points[1][1] - points[0][1]), 2)
    secondline_length = math.pow((points[2][0] - points[1][0]), 2) + math.pow((points[2][1] - points[1][1]), 2)

    if firstline_length > secondline_length:
        return __calculate_line_degree(points[0], points[1])
    else:
        return __calculate_line_degree(points[2], points[1])


def __get_rrect_area(_rrect):
    """
    Get the area of the rotate rect
    :param _rrect:
    :return:
    """
    points = cv2.boxPoints(box=_rrect)
    firstline_length = math.sqrt(math.pow((points[1][0] - points[0][0]), 2) +
                                 math.pow((points[1][1] - points[0][1]), 2))
    secondline_length = math.sqrt(math.pow((points[2][0] - points[1][0]), 2) +
                                  math.pow((points[2][1] - points[1][1]), 2))
    return firstline_length*secondline_length


def __is_rrect_valid(rrect):
    """
    Thresh the invalid rotate rect through the angle and area
    :param rrect:
    :return:
    """
    rrect_angle = __get_rrect_degree(rrect)
    if rrect_angle < 45 or rrect_angle > 135:
        return False
    #
    rrect_area = __get_rrect_area(rrect)
    if rrect_area < 12*12:
        return False
    return True


def __map_roi_to_front_view(roidb):
    """
    Map the roidb to the front view image through perspective mapping function
    :param roidb: top view roidb
    :return: front view roidb , if the converted front view roidb's bndbox or contours is invalid (mainly because the
    mapped points on the front view image may be out of the image boundry) the return false as the roi flag to show this
    roi is a invalid roi that can't compose a roi pair
    """
    top_roi_index = roidb.get_roi_index()
    top_roi_contours = roidb.get_roi_contours()
    top_roi_response_points = roidb.get_roi_response_points()

    roidb_is_valid = True
    fv_roi_contours = []
    fv_roi_response_points = []

    for index, point in enumerate(top_roi_contours):
        # map the point from top crop image to top image
        pt1 = [point[0]+START_X, point[1]+START_Y]
        fv_point = perspective_point(pt1=pt1)
        if fv_point[0] < 0 or fv_point[0] >= WARPED_IMAGE_WIDTH or fv_point[1] < 0 \
                or fv_point[1] >= WARPED_IMAGE_HEIGHT:
            roidb_is_valid = False
            break
        fv_roi_contours.append(fv_point)

    for index, points in enumerate(top_roi_response_points):
        # map the point from top crop image to top image
        pt1 = [point[0] + START_X, point[1] + START_Y]
        fv_point = perspective_point(pt1=pt1)
        if fv_point[0] < 0 or fv_point[0] >= WARPED_IMAGE_WIDTH or fv_point[1] < 0 \
                or fv_point[1] >= WARPED_IMAGE_HEIGHT:
            roidb_is_valid = False
            break
        fv_roi_response_points.append(fv_point)
    fv_roi_contours = np.array(fv_roi_contours)
    fv_roi_response_points = np.array(fv_roi_contours)
    fv_roi = Roidb(roi_index=top_roi_index, roi_contours=fv_roi_contours, roi_response_points=fv_roi_response_points)
    return fv_roi, roidb_is_valid


def __find_response_points_in_contours(contours, image):
    """
    find responding points in contours' bndbox and responding points are those points with value 255 in the OTSU result
    of weight hat like filtered image
    :param contours:
    :param image: OTSU threshold image
    :return:
    """
    assert len(contours) > 0

    result = []

    for index, contour in enumerate(contours):
        bndbox = cv2.boundingRect(contour)
        roi = image[bndbox[1]:bndbox[1]+bndbox[3], bndbox[0]:bndbox[0]+bndbox[2]]
        response_points = np.vstack((np.where(np.array(roi) == 255)[1],
                                     np.where(np.array(roi) == 255)[0])).T
        response_points[:, 0] += bndbox[0]
        response_points[:, 1] += bndbox[1]
        result.append(response_points)
    return np.array(result)


def binarized_whatlike_filtered_image(img):
    """
    Do normalization and thresholding on the result of weighted hat-like filter image to extract line candidate
    :param img: input image
    :return: list of roi pair (top_roi, fv_roi) class which defined in imdb.py
    """
    if img is None:
        raise ValueError('Image data is invalid')
    # min max normalize the image
    image = img[:, :, 0]
    image = np.uint8(image)
    norm_image = np.zeros(image.shape)
    norm_image = cv2.normalize(src=image, dst=norm_image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # otsu thresh image
    blur = cv2.GaussianBlur(src=norm_image, ksize=(5, 5), sigmaX=0, sigmaY=0)
    ret, thresh_image = cv2.threshold(src=blur, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # find connected component
    image, contours, hierarchy = cv2.findContours(image=thresh_image, mode=cv2.RETR_EXTERNAL,
                                                  method=cv2.CHAIN_APPROX_SIMPLE)
    response_points = __find_response_points_in_contours(contours=contours, image=thresh_image)
    # find rotate rect of each contour and check if it fits the condition, if fits the condition then save the bounding
    # rectangle of the contour
    result = []
    valid_contours = 0
    for index, contour in enumerate(contours):
        rotrect = cv2.minAreaRect(contour)
        if __is_rrect_valid(rotrect):
            # the contours is valid and can be saved
            roi_contours = contour
            roi_contours = np.reshape(roi_contours, newshape=(roi_contours.shape[0], roi_contours.shape[2]))
            roi_index = valid_contours
            valid_contours += 1
            top_roi_db = Roidb(roi_index=roi_index, roi_contours=roi_contours,
                               roi_response_points=response_points[index])  # type:
            fv_roi_db, roi_is_valid = __map_roi_to_front_view(roidb=top_roi_db)
            if roi_is_valid:
                result.append((top_roi_db, fv_roi_db))
    return result
