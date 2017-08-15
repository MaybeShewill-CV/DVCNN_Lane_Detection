"""
Detect lane line from front view image and top view image
"""
import os
import os.path as ops
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from dvcnn_classifier import DVCNNClassifier
from extract_roi import extract_roi_candidates
from global_optimization import Optimizer

START_X = 325
START_Y = 327
CROP_WIDTH = 325
CROP_HEIGHT = 325


def detect_lane_line(top_view_image_file, front_view_image_file, model_file, weights_file):
    """
    Detect lane line from top view image and front view image
    :param top_view_image_file:
    :param front_view_image_file:
    :param model_file:
    :param weights_file:
    :return:
    """
    if not ops.isfile(top_view_image_file) or not ops.isfile(front_view_image_file):
        raise ValueError('{:s} or {:s} is invalid file'.format(top_view_image_file, front_view_image_file))

    top_image = cv2.imread(top_view_image_file, cv2.IMREAD_UNCHANGED)
    fv_image = cv2.imread(front_view_image_file, cv2.IMREAD_UNCHANGED)

    assert top_image is not None
    assert fv_image is not None

    # crop top image patch because the bottom part of the perspective top image from front image is largely variant
    top_image_patch = top_image[START_Y:START_Y+CROP_HEIGHT, START_X:START_X+CROP_WIDTH, :]
    top_image_patch_copy = top_image_patch.copy()
    top_image_patch_copy2 = np.zeros(shape=(350, 350), dtype=np.uint8)

    # extract roi candidates from top image
    roi_pairs, filtered_image = extract_roi_candidates(image=top_image_patch)

    # use dvcnn to classify the roi is a lane line or not
    top_rois = []
    fv_rois = []
    for index, roi in enumerate(roi_pairs):
        top_roi = roi[0]
        fv_roi = roi[1]
        top_roi_bndbox = top_roi.get_roi_bndbox()
        fv_roi_bndbox = fv_roi.get_roi_bndbox()
        top_rois_image = top_image_patch[top_roi_bndbox[1]:top_roi_bndbox[1]+top_roi_bndbox[3],
                                         top_roi_bndbox[0]:top_roi_bndbox[0]+top_roi_bndbox[2],
                                         :]
        top_rois_image = cv2.resize(src=top_rois_image, dsize=(64, 64))
        top_rois.append(top_rois_image)
        fv_rois_image = fv_image[fv_roi_bndbox[1]:fv_roi_bndbox[1]+fv_roi_bndbox[3],
                                 fv_roi_bndbox[0]:fv_roi_bndbox[0]+fv_roi_bndbox[2],
                                 :]
        fv_rois_image = cv2.resize(src=fv_rois_image, dsize=(128, 128))
        fv_rois.append(fv_rois_image)

    dvcnnclassifier = DVCNNClassifier(model_file=model_file, weights_file=weights_file)
    predictions = np.array(dvcnnclassifier.predict(top_view_image_list=top_rois, front_view_image_list=fv_rois))
    scores = predictions[:, 1]
    # set the roi scores attribute
    for index, roi in enumerate(roi_pairs):
        top_roi = roi[0]
        fv_roi = roi[1]
        top_roi.set_roi_dvcnn_score(scores[index])
        fv_roi.set_roi_dvcnn_score(scores[index])

    inds = np.where(np.all([predictions[:, 0] == 1, predictions[:, 1] >= 0.0], axis=0))[0]
    # inds = np.where(predictions[:, 1] >= 0.0)[0]

    # select the remain roi pairs after dvcnn detect
    remain_roi_pairs = []
    for index in inds:
        remain_roi_pairs.append(roi_pairs[index])

    if len(remain_roi_pairs) == 0:
        print('No lane line rois are detected in this image pair')
        return

    # visualize the remain rois
    for index, roi in enumerate(remain_roi_pairs):
        top_roi = roi[0]
        fv_roi = roi[1]
        top_roi_bndbox = top_roi.get_roi_bndbox()
        fv_roi_bndbox = fv_roi.get_roi_bndbox()
        top_roi_dvcnn_score = top_roi.get_roi_dvcnn_score()
        fv_roi_dvcnn_score = fv_roi.get_roi_dvcnn_score()
        # draw top view image rois
        cv2.rectangle(top_image_patch, (top_roi_bndbox[0], top_roi_bndbox[1]),  # pt1
                      (top_roi_bndbox[0] + top_roi_bndbox[2], top_roi_bndbox[1] + top_roi_bndbox[3]),  # pt2
                      (0, 255, 0),  # color
                      2)  # thickness
        # write the dvcnn score
        cv2.putText(top_image_patch, '{:5f}'.format(top_roi_dvcnn_score), (top_roi_bndbox[0]+5, top_roi_bndbox[1]+5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        # draw front view image rois
        cv2.rectangle(fv_image, (fv_roi_bndbox[0], fv_roi_bndbox[1]),  # pt1
                      (fv_roi_bndbox[0] + fv_roi_bndbox[2], fv_roi_bndbox[1] + fv_roi_bndbox[3]),  # pt2
                      (0, 255, 0),  # color
                      2)  # thickness
        cv2.putText(fv_image, '{:5f}'.format(fv_roi_dvcnn_score), (fv_roi_bndbox[0]+10, fv_roi_bndbox[1]+10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    plt.figure('Front view with DVCNN score')
    plt.imshow(fv_image[:, :, (2, 1, 0)])
    plt.figure('Top view with DVCNN score')
    plt.imshow(top_image_patch[:, :, (2, 1, 0)])
    # plt.figure('Weight hat like filter image')
    # plt.imshow(filtered_image[:, :, 0], cmap='gray')

    optimizer = Optimizer(roidb_pair_list=remain_roi_pairs)
    merged_roi_list = optimizer.calculate_rois_score()
    # visualize the line in top view image
    merged_roi_list = np.array(merged_roi_list)
    sort_index = np.argsort(merged_roi_list[:, (1, 0)][:, 0], axis=0)[::-1]
    sorted_roi_list = merged_roi_list[:, (1, 0)][tuple(sort_index), :]
    count_index = 0
    for index, roi in enumerate(sorted_roi_list):
        roi_score = roi[0]
        # if roi_score < 0.75:
        #     count_index += 1
        #     break
        [vx, vy, x, y] = roi[1].get_roi_line_param()
        contours = roi[1].get_roi_contours()
        response_points = roi[1].get_roi_response_points()
        roi_x = response_points[:, 0][np.argsort(response_points[:, 0], axis=0)]
        roi_y = response_points[:, 1][np.argsort(response_points[:, 1], axis=0)]

        for point in response_points:
            top_image_patch_copy2[point[1], point[0]] = 255

        roi_x_strictly_increase = []
        roi_y_strictly_increase = []
        for idx, x in enumerate(roi_x):
            if np.count_nonzero(a=x) >= 1:
                if roi_x_strictly_increase.count(x) >= 1:
                    continue
                else:
                    roi_x_strictly_increase.append(x)
                    roi_y_strictly_increase.append(roi_y[idx])
        bsp = BSpline(np.array(roi_x_strictly_increase), np.array(roi_y_strictly_increase), k=1)
        lefty = int((-x * vy / vx) + y)
        righty = int(((top_image_patch.shape[1] - x) * vy / vx) + y)
        pt1 = (top_image_patch.shape[1] - 1, righty)
        pt2 = (0, lefty)
        cv2.line(top_image_patch_copy, pt1, pt2, (255, 0, 255), 2)
        cv2.putText(top_image_patch_copy, '{:5f}'.format(roi_score), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        # xp = np.linspace(0, 349, 350)
        # yp = bsp(xp)
        # new_contours = np.vstack((xp, yp)).T
        # new_contours_copy = []
        # for points in new_contours:
        #     x = points[0]
        #     y = points[1]
        #     if x < 0 or x > 350 or y < 0 or y > 350:
        #         continue
        #     else:
        #         new_contours_copy.append(points)
        # new_contours_copy = np.array(new_contours_copy).astype(np.int32)
        # cv2.polylines(img=top_image_patch_copy, pts=[new_contours_copy], isClosed=False, color=(0, 255, 0))
        # if count_index > 1:
        #     break
        count_index += 1
    # plt.figure('Top view with final lane line and optimization score')
    # plt.imshow(top_image_patch_copy[:, :, (2, 1, 0)])
    # plt.figure('Fuck')
    # plt.imshow(top_image_patch_copy2, cmap='gray')
    plt.show()
    return


def main(top_view_dir):
    top_file_list = []
    for parents, _, filenames in os.walk(top_view_dir):
        for filename in filenames:
            top_file_list.append(ops.join(parents, filename))

    for index, top_file in enumerate(top_file_list):
        [top_file_dir, top_file_id] = ops.split(top_file)
        fv_file_dir = top_file_dir.replace('top', 'fv')
        fv_file_id = top_file_id.replace('top', 'fv')
        fv_file = ops.join(fv_file_dir, fv_file_id)

        detect_lane_line(top_view_image_file=top_file, front_view_image_file=fv_file,
                         model_file='DVCNN/model_def/DVCNN.json', weights_file='DVCNN/model/dvcnn.ckpt-1199')
    return


if __name__ == '__main__':
    # detect_lane_line(top_view_image_file='data/top_view/test_top.jpg',
    #                  front_view_image_file='data/front_view/test_fv.jpg',
    #                  model_file='DVCNN/model_def/DVCNN.json',
    #                  weights_file='DVCNN/model/dvcnn.ckpt-1199')
    # print('Done')
    main(top_view_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/Origin/top_view')
