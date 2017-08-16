"""
Walk through the lane line front view samples folder and select their corresponding top view samples. Get rid of them
from the whole front and top view images to get the negative training samples
"""
import os
import shutil
import sys
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


class Samples(object):
    def __init__(self, fv_samples_dir, top_samples_dir, lane_fv_samples_dir):
        """
        Initialization
        :param fv_samples_dir: front view roi files dir
        :param top_samples_dir: top view roi files dir
        :param lane_fv_samples_dir: front view lane roi files dir
        """
        if os.path.exists(fv_samples_dir):
            self.fv_samples_dir = fv_samples_dir
        else:
            raise ValueError('{:s} doesn\'t exist'.format(fv_samples_dir))
        if os.path.exists(top_samples_dir):
            self.top_samples_dir = top_samples_dir
        else:
            raise ValueError('{:s} doesn\'t exist'.format(top_samples_dir))
        if os.path.exists(lane_fv_samples_dir):
            self.lane_fv_samples_dir = lane_fv_samples_dir
        else:
            raise ValueError('{:s} doesn\'t exist'.format(lane_fv_samples_dir))

        self.lane_fv_samples = self.__get_filename(lane_fv_samples_dir)
        self.fv_samples = self.__get_filename(fv_samples_dir)
        self.top_samples = self.__get_filename(top_samples_dir)

        self.lane_top_samples = self.__select_lane_top_samples()
        # self.non_lane_top_samples = self.__select_non_lane_top_samples()
        self.non_lane_fv_samples = self.__select_non_lane_fv_samples()
        self.non_lane_top_samples = self.__select_non_lane_top_samples()

    @staticmethod
    def __get_filename(src_dir):
        """
        Get the whole file name from src _dir
        :param src_dir:
        :return:
        """
        result = []
        for parents, dirnames, filenames in os.walk(src_dir):
            for filename in filenames:
                result.append(filename)
        return result

    def __select_lane_top_samples(self):
        """
        select the corresponding lane top view samples
        :return:
        """
        result = []
        for index, fv_image_id in enumerate(self.lane_fv_samples):
            image_id_prefix = fv_image_id.replace('fv', 'top')
            top_image_id = ''
            for top_index, top_image_id_tmp in enumerate(self.top_samples):
                if top_image_id_tmp == image_id_prefix:
                    top_image_id = top_image_id_tmp
                    result.append(top_image_id)
                    del self.top_samples[top_index]
                    break
            if top_image_id == '':
                raise ValueError('Can\'t find {:s} in lane top image file list'.format(image_id_prefix))
            sys.stdout.write('\r>>Selecting lane top samples {:d}/{:d} {:s}'.format(index+1,
                                                                                    len(self.lane_fv_samples),
                                                                                    top_image_id))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        return result

    def __select_non_lane_fv_samples(self):
        """
        Select non lane front view samples by getting rid of the lane fv samples from the total fv samples
        :return:
        """
        result = []
        for index, fv_image_id in enumerate(self.fv_samples):
            is_lane = False
            for fv_index, fv_lane_image_id in enumerate(self.lane_fv_samples):
                if fv_lane_image_id == fv_image_id:
                    is_lane = True
                    del self.lane_fv_samples[fv_index]
                    break
            if not is_lane:
                result.append(fv_image_id)
            sys.stdout.write('\r>>Selecting non lane fv sample {:d}/{:d} {:s}'.
                             format(index+1, len(self.fv_samples) - len(self.lane_fv_samples), fv_image_id))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        return result

    def __select_non_lane_top_samples(self):
        """
        Select non lane top view samples by getting rid of the lane top samples from the total top samples
        :return:
        """
        result = []
        for index, fv_image_id in enumerate(self.non_lane_fv_samples):
            image_id_prefix = fv_image_id.replace('fv', 'top')
            top_image_id = ''
            for top_index, top_image_id_tmp in enumerate(self.top_samples):
                if top_image_id_tmp == image_id_prefix:
                    top_image_id = top_image_id_tmp
                    result.append(top_image_id)
                    del self.top_samples[top_index]
                    break
            if top_image_id == '':
                raise ValueError('Can\'t find {:s} in top image file list'.format(image_id_prefix))
            sys.stdout.write('\r>>Selecting non lane top samples {:d}/{:d} {:s}'.format(index+1,
                                                                                        len(self.non_lane_fv_samples),
                                                                                        top_image_id))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        return result

    def generate_samples(self, lane_top_samples_dir, non_lane_fv_images_dir, non_lane_top_images_dir):
        """
        generate training samples
        :return:
        """
        if not os.path.exists(lane_top_samples_dir):
            print('{:s} doen\'t exist and has been created the first time'.format(lane_top_samples_dir))
            os.makedirs(lane_top_samples_dir)
        if not os.path.exists(non_lane_fv_images_dir):
            print('{:s} doen\'t exist and has been created the first time'.format(non_lane_fv_images_dir))
            os.makedirs(non_lane_fv_images_dir)
        if not os.path.exists(non_lane_top_images_dir):
            print('{:s} doen\'t exist and has been created the first time'.format(non_lane_top_images_dir))
            os.makedirs(non_lane_top_images_dir)

        # generate lane top samples
        print('Start generating lane top view samples')
        for index, top_lane_image_id in enumerate(self.lane_top_samples):
            src_top_lane_image_file = os.path.join(self.top_samples_dir, top_lane_image_id)
            if not os.path.isfile(src_top_lane_image_file):
                raise ValueError('{:s} doesn\'t exist'.format(src_top_lane_image_file))
            dst_top_lane_image_file = os.path.join(lane_top_samples_dir, top_lane_image_id)
            # copy file can be changed into move file
            shutil.copy(src_top_lane_image_file, dst_top_lane_image_file)
            sys.stdout.write('\r>>Copying lane top samples {:d}/{:d} {:s}'.format(index+1, len(self.lane_top_samples),
                                                                                  top_lane_image_id))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

        # generate non lane front view samples
        print('Start generating non lane front view samples')
        stop_index = 0
        for index, non_lane_fv_image_id in enumerate(self.non_lane_fv_samples):
            src_fv_non_lane_image_file = os.path.join(self.fv_samples_dir, non_lane_fv_image_id)
            if not os.path.isfile(src_fv_non_lane_image_file):
                raise ValueError('{:s} doesn\'t exist'.format(src_fv_non_lane_image_file))
            dst_fv_non_lane_image_file = os.path.join(non_lane_fv_images_dir, non_lane_fv_image_id)
            shutil.copy(src_fv_non_lane_image_file, dst_fv_non_lane_image_file)
            if stop_index >= 3000:
                break
            sys.stdout.write('\r>>Copying non lane front samples {:d}/{:d} {:s}'.format(index + 1,
                                                                                        3000,
                                                                                        non_lane_fv_image_id))
            sys.stdout.flush()
            stop_index += 1
        sys.stdout.write('\n')
        sys.stdout.flush()

        # generate non lane top view samples
        print('Start generating non lane top view samples')
        stop_index = 0
        for index, non_lane_top_images_id in enumerate(self.non_lane_top_samples):
            src_top_non_lane_image_file = os.path.join(self.top_samples_dir, non_lane_top_images_id)
            if not os.path.isfile(src_top_non_lane_image_file):
                raise ValueError('{:s} doesn\'t exist'.format(src_top_non_lane_image_file))
            dst_top_non_lane_image_file = os.path.join(non_lane_top_images_dir, non_lane_top_images_id)
            shutil.copy(src_top_non_lane_image_file, dst_top_non_lane_image_file)
            if stop_index >= 3000:
                break
            sys.stdout.write('\r>>Copying non lane top samples {:d}/{:d} {:s}'.format(index + 1,
                                                                                      3000,
                                                                                      non_lane_top_images_id))
            sys.stdout.flush()
            stop_index += 1
        sys.stdout.write('\n')
        sys.stdout.flush()
        print('Generating dvcnn samples complete')
        return


def pick_up_valid_sample(top_view_dir, front_view_dir):
    """
    pick the valid pair of two images in top view and front view folder
    :param top_view_dir:
    :param front_view_dir:
    :return:
    """
    fv_image_list = []
    for parents, _, filenames in os.walk(front_view_dir):
        for filename in filenames:
            fv_image_list.append(filename)

    top_image_list = []
    for parents, _, filenames in os.walk(top_view_dir):
        for filename in filenames:
            top_image_list.append(filename)

    for index, fv_image_id in enumerate(fv_image_list):
        image_id_prefix = fv_image_id.replace('fv', 'top')
        for top_image_id_tmp in top_image_list:
            if top_image_id_tmp == image_id_prefix:
                fv_image = cv2.imread(os.path.join(front_view_dir, fv_image_id), cv2.IMREAD_UNCHANGED)
                top_image = cv2.imread(os.path.join(top_view_dir, top_image_id_tmp), cv2.IMREAD_UNCHANGED)
                if fv_image is not None and top_image is not None:
                    break
                else:
                    os.remove(os.path.join(front_view_dir, fv_image_id))
                    os.remove(os.path.join(top_view_dir, top_image_id_tmp))
                    break
        else:
            continue
        sys.stdout.write('\r>>Picking up {:d}/{:d} {:s}'.format(index+1, len(fv_image_list), fv_image_id))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    return


def main():
    dvcnn_samples = Samples(fv_samples_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE_TEST/'
                                           'lane_line/front_view',
                            top_samples_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE_TEST'
                                            '/lane_line/top_view',
                            lane_fv_samples_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/DVCNN_SAMPLE/'
                                                'lane_line/front_view')
    dvcnn_samples.generate_samples(lane_top_samples_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                                        'DVCNN_SAMPLE/lane_line/top_view',
                                   non_lane_fv_images_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                                          'DVCNN_SAMPLE/non_lane_line/front_view',
                                   non_lane_top_images_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                                           'DVCNN_SAMPLE/non_lane_line/top_view')
    return 1

if __name__ == '__main__':
    main()
    pick_up_valid_sample(top_view_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                      'DVCNN_SAMPLE/lane_line/top_view',
                         front_view_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                        'DVCNN_SAMPLE/lane_line/front_view')
    pick_up_valid_sample(top_view_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                      'DVCNN_SAMPLE/non_lane_line/top_view',
                         front_view_dir='/home/baidu/DataBase/Road_Center_Line_DataBase/'
                                        'DVCNN_SAMPLE/non_lane_line/front_view')
    print('Done')
