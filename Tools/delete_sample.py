"""
Delete the front sample and its' corresponding top sample which listed in lane fv sample dir
"""
import os
import sys
import time


def delete_sample(lane_fv_sample_dir, fv_sample_dir, top_sample_dir):
    """
    delete the front sample and its' corresponding top sample which listed in lane fv sample dir
    :param lane_fv_sample_dir:
    :param fv_sample_dir:
    :param top_sample_dir:
    :return:
    """
    for parents, _, filenames in os.walk(lane_fv_sample_dir):
        for index, filename in enumerate(filenames):
            fv_file_path = os.path.join(fv_sample_dir, filename)
            top_file_path = os.path.join(top_sample_dir, filename.replace('fv', 'top'))
            if os.path.isfile(fv_file_path) and os.path.isfile(top_file_path):
                os.remove(fv_file_path)
                os.remove(top_file_path)
            if os.path.isfile(fv_file_path) and not os.path.isfile(top_file_path):
                raise ValueError('{:s} doesn\'t exist'.format(top_file_path))
            if not os.path.isfile(fv_file_path) and os.path.isfile(top_file_path):
                raise ValueError('{:s} doesn\'t exist'.format(fv_file_path))
            sys.stdout.write('\r>>Removing {:d}/{:d} {:s}'.format(index+1, len(filenames), filename))
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    return


if __name__ == '__main__':
    t_start = time.time()
    delete_sample(lane_fv_sample_dir='front_view_non_lane_line_for_training',
                  fv_sample_dir='front_view',
                  top_sample_dir='top_view')
    delete_sample(lane_fv_sample_dir='front_view_lane_line_for_training',
                  fv_sample_dir='front_view',
                  top_sample_dir='top_view')
    print('Elapsed time: {:5f}s'.format(time.time() - t_start))
