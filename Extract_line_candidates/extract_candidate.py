import math
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
try:
    from cv2 import cv2
except ImportError:
    pass

import dataset_util
import filter_util


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


def __get_rrect_ratio(_rrect):
    """
    Calculate the ratio between the long side and the short side of the rotate rect
    :param __rrect:
    :return:
    """
    points = cv2.boxPoints(box=_rrect)
    firstline_length = math.pow((points[1][0] - points[0][0]), 2) + math.pow((points[1][1] - points[0][1]), 2)
    secondline_length = math.pow((points[2][0] - points[1][0]), 2) + math.pow((points[2][1] - points[1][1]), 2)

    ratio = np.amax([firstline_length, secondline_length]) / np.amin([firstline_length, secondline_length])
    return ratio


def __get_rrect_aera(_rrect):
    points = cv2.boxPoints(box=_rrect)
    firstline_length = math.sqrt(math.pow((points[1][0] - points[0][0]), 2) +
                                 math.pow((points[1][1] - points[0][1]), 2))
    secondline_length = math.sqrt(math.pow((points[2][0] - points[1][0]), 2) +
                                  math.pow((points[2][1] - points[1][1]), 2))
    return firstline_length*secondline_length


def __is_rrect_valid(rrect):
    rrect_angle = __get_rrect_degree(rrect)
    if rrect_angle < 45 or rrect_angle > 135:
        return False
    #
    rrect_area = __get_rrect_aera(rrect)
    if rrect_area < 12*12:
        return False
    #
    # rrect_ratio = __get_rrect_ratio(rrect)
    # if rrect_ratio < 2.5:
    #     return False
    return True


def __extract_line_from_filtered_image(img):
    """
    Do normalization and thresholding on the result of weighted hat-like filter image to extract line candidate
    :param img:input image
    :return:rotate rect list []
    """
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

    # find rotate rect of each contour and check if it fits the condition, if fits the condition then save the bounding
    # rectangle of the contour
    rotate_rect_list = []
    bounding_rect_list = []
    for i in range(len(contours)):
        contour = contours[i]
        rotrect = cv2.minAreaRect(contour)
        if __is_rrect_valid(rotrect):
            rotate_rect_list.append(rotrect)
            bnd_rect = cv2.boundingRect(contour)
            bounding_rect_list.append(bnd_rect)
    result = {
        'rotate_rect_list': rotate_rect_list,
        'bounding_rect_list': bounding_rect_list
    }
    return result


def extract_line_candidates(image_file_dir, image_flag):
    """
    Use hat-like filter, normalization and thresholding to extract line candidates in image_file_dir
    :param image_file_dir: image file store path
    :param image_flag: image type
    :return: candidate_result image_file_list. The candidate_result[i] is correspond to image_file_list[i]
    """
    dataset = dataset_util.Dataset(image_file_dir, image_flag)
    image_file_list = dataset.get_filelist()
    image_file_nums = dataset.get_filenums()
    whatlikefilter = filter_util.WHatLikeFilter([9, 4])

    image_filename_queue = tf.train.string_input_producer(image_file_list, shuffle=False)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(queue=image_filename_queue)

    image = tf.image.decode_jpeg(contents=image_file, channels=1)
    image.set_shape((325, 325, 1))

    image_batch = tf.train.batch(tensors=[image], batch_size=128, num_threads=1)

    rotrect_candidatas_list = []
    bndrect_candidates_list = []

    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        epochs = math.ceil(image_file_nums / 128)
        for i in range(epochs):
            # apply weight hat-like filter function
            filter_result = sess.run(whatlikefilter.filter(image_batch))
            for filter_image in filter_result:
                # apply image threshold and components extraction
                rotrect_candidatas_list.append(__extract_line_from_filtered_image(filter_image)['rotate_rect_list'])
                bndrect_candidates_list.append(__extract_line_from_filtered_image(filter_image)['bounding_rect_list'])

        coord.request_stop()
        coord.join(threads=threads)

    result = {
        'rotate_rect_candidates': rotrect_candidatas_list,
        'bounding_rect_candidates': bndrect_candidates_list,
        'image_file_list': image_file_list
    }

    return result


def extract_all(top_file_dir, is_vis=False, vis_result_save_path=None):
    if not os.path.exists(top_file_dir):
        raise ValueError('{:s} doesn\'t exist'.format(top_file_dir))
    if is_vis and vis_result_save_path is None:
        raise ValueError("You should supply the vis_result_save_path")
    if is_vis and not os.path.exists(vis_result_save_path):
        print('{:s} doesn\'t exist and has been created'.format(vis_result_save_path))
        os.makedirs(vis_result_save_path)

    # file in res[i] and image_file_list[i] matches each other
    result = extract_line_candidates(top_file_dir, 'jpg')
    image_file_list = result['image_file_list']
    rrect_list_all = result['rotate_rect_candidates']
    bndrect_list_all = result['bounding_rect_candidates']

    res_info_dict = dict()

    for i, filename in enumerate(image_file_list):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        [_, image_id] = os.path.split(filename)
        rrect_list = rrect_list_all[i]  # rotate rects in image_file_list[i] got from res[i]
        bndrect_list = bndrect_list_all[i]

        res_info_dict[image_id] = dict()
        res_info_dict[image_id]['rrect'] = rrect_list
        res_info_dict[image_id]['bndrect'] = bndrect_list

        if is_vis:
            for j, rrect in enumerate(rrect_list):
                rotbox = cv2.boxPoints(rrect)
                rotbox = np.int0(rotbox)
                cv2.drawContours(image, [rotbox], 0, (0, 0, 255), 2)
                bndbox = bndrect_list[j]
                cv2.rectangle(image, (bndbox[0], bndbox[1]),  # pt1
                              (bndbox[0] + bndbox[2], bndbox[1] + bndbox[3]),  # pt2
                              (0, 255, 0),  # color
                              0)  # thickness
            res_save_path = os.path.join(vis_result_save_path, image_id)
            cv2.imwrite(res_save_path, image)
            print('Extrct {:s} done with vis'.format(image_id))
        else:
            print('Extrct {:s} done without vis'.format(image_id))
    print('Extraction complete!')
    return res_info_dict


# if __name__ == '__main__':
#     parse = argparse.ArgumentParser(description='Extract line candidates from top view images')
#     parse.add_argument('--top_file_dir', type=str, default=None, help='Where you store the top view images')
#     parse.add_argument('--extract_save_dir', type=str, default=None, help='Where you store the candidates result')
#
#     args = parse.parse_args()
#
#     if not args.top_file_dir:
#         raise ValueError('You must supply top view file dir with --top_file_dir')
#     if not args.extract_save_dir:
#         raise ValueError('You must supply result storage path with --extract_save_dir')
#
#     res_info = extract_all(args.top_file_dir, args.extract_save_dir)
if __name__ == '__main__':
    extract_line_candidates(image_file_dir='data/top_view', image_flag='jpg')
