import sys
import glob
import os
import logging
import re

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


def _re(path):
    return re.findall(r"^([0-9]{6})_([CLR])_\(([-.0-9]+), ([-.0-9]+)", path)[0]


def read(path, steering_normalizer):
    """Read driving_log from image names"""

    sub_dirs = [x[0] + '/' for x in os.walk(path)][1:]

    logging.info('Found {} directories'.format(len(sub_dirs)))

    ret_dict = {}
    for sdir in sub_dirs:
        im_paths = glob.glob(sdir + '*.jpeg')
        im_paths_slices = [os.path.splitext(os.path.basename(im))[0] for im in im_paths]

        for i in range(len(im_paths)):
            ID, pos, steer, thorttle = _re(im_paths_slices[i])

            if ID not in ret_dict:
                ret_dict[ID] = {}

            steer = float(steer) / steering_normalizer
            ret_dict[ID][pos] = (im_paths[i], steer, thorttle)

    logging.info('Found {} image triplets (L, C, R)'.format(len(ret_dict)))

    return ret_dict


if __name__ == "__main__":
    path = sys.argv[1]
    path_dict = read(path)
