"""Create driving log from image filenames."""

import sys
import glob
import os
import logging
import re
import math

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def _re(path):
    """Regex to find necesarry information from filename."""

    return re.findall(r"^([0-9]{6})_([CLR])_\(([-.0-9]+), ([-.0-9]+)", path)[0]

def read(path):
    """Read driving_log from image names"""

    sub_dirs = [x[0] + '/' for x in os.walk(path)][1:]

    logging.info('Found %d directories', len(sub_dirs))

    ret_dict = {}
    for sdir in sub_dirs:
        im_paths = glob.glob(sdir + '*.jpeg')
        im_paths_slices = [os.path.splitext(os.path.basename(im))[0] for im in im_paths]

        for i, im_p in enumerate(im_paths):
            id_nr, pos, steer, thorttle = _re(im_paths_slices[i])

            if id_nr not in ret_dict:
                ret_dict[id_nr] = {}

            ret_dict[id_nr][pos] = (im_p, float(steer), thorttle)

    logging.info('Found %d image triplets (L, C, R)', len(ret_dict))

    ret = [item[1] for item in ret_dict.items()]

    return ret


if __name__ == "__main__":
    PATH = sys.argv[1]
    PATH_DICT = read(PATH)
