"""
Record the origin file name and rename the file name in order to find it's corresponding file
"""
import os
import sys

fv_name_file = open('top_filename.txt', 'w')


def find_substr_itimes(_str, _substr, i):
    """
    find the location of the substr appered i times in str
    :param _str:
    :param _substr:
    :param i:
    :return:
    """
    count = 0
    while i > 0:
        index = _str.find(_substr)
        if index == -1:
            return -1
        else:
            _str = _str[index+1:]
            i -= 1
            count = count + index + 1
    return count - 1

for parents, _, filenames in os.walk('top_view'):
    for index, filename in enumerate(filenames):
        fv_name_file.write(filename + '\n')
        new_filename = '{}.jpg'.format(filename[0:find_substr_itimes(filename, '_', 3)])
        os.rename(src=os.path.join(parents, filename), dst=os.path.join(parents, new_filename))
        sys.stdout.write('\r>>Record {:d}/{:d} {:s}'.format(index, len(filenames), new_filename))
        sys.stdout.flush()
sys.stdout.write('\n')
sys.stdout.flush()
