"""
Crop top view image to get a roi image, record the crop start position in order to be able to remap the points in the
roi image to front view image
"""
import os
import sys
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

START_X = 325
START_Y = 327
CROP_WIDTH = 325
CROP_HEIGHT = 325


def crop_image(src, start_x, start_y, width, height):
    [src_height, src_width] = src.shape[:-1]
    assert (start_y + height) < src_height
    assert (start_x + width) < src_width

    if len(src.shape) == 2:  # with 1 channel
        return src[start_y:start_y+height, start_x:start_x+width]
    elif len(src.shape) == 3:
        return src[start_y:start_y+height, start_x:start_x+width, :]
    else:
        raise ValueError('Invalid image shape')


def main():
    top_image_path = '/home/baidu/DataBase/Road_Center_Line_DataBase/Origin/top_view'
    crop_image_save_path = '/home/baidu/DataBase/Road_Center_Line_DataBase/Origin/top_view_crop'

    if not os.path.exists(top_image_path):
        raise ValueError('{:s} doesn\'t exist'.format(top_image_path))
    if not os.path.exists(crop_image_save_path):
        os.makedirs(crop_image_save_path)

    for parents, dirnames, filenames in os.walk(top_image_path):
        for index, filename in enumerate(filenames):
            top_image_filename = os.path.join(parents, filename)
            top_image = cv2.imread(top_image_filename, cv2.IMREAD_UNCHANGED)
            top_crop_image = crop_image(src=top_image, start_x=START_X, start_y=START_Y, width=CROP_WIDTH,
                                        height=CROP_HEIGHT)
            crop_roi_save_path = os.path.join(crop_image_save_path, filename)
            cv2.imwrite(crop_roi_save_path, top_crop_image)
            sys.stdout.write('\r>>Map {:d}/{:d} {:s}'.format(index + 1, len(filenames), filename))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
    return


if __name__ == '__main__':
    main()
