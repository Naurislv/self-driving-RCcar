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

def angle2tcr(angle, wheelbase, center_of_mass=0):
    """Convert cars steering angle to curvatrue which is  1 / Turning Circle Radius.

    Inputs:
        angle: steering angle in degrees
        wheelbase: distance between wheels contact points in mm
        center_of_mass: distance between the back wheel contact point and centor of mass in mm

    Output:
        curvature: Inverse Turning Circle radius or curvature

    Resources:
        https://goo.gl/FNf8CZ
    """

    # Find TCR using Bycicle Model
    # https://sites.google.com/site/bikephysics/english-version/2-geometry-and-kinematics

    # If we don't know center_of mass then we assue it is 0
    if center_of_mass == 0:
        curvature = math.tan(math.radians(angle)) / wheelbase
    else:
        angle += 1e-12  # avoid dealing with 0
        curvature = 1 / math.sqrt(center_of_mass**2 +
                                  (wheelbase / math.tan(math.radians(angle)))**2)

    return curvature

def read(path, wheelbase):
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

            tcr = angle2tcr(float(steer), wheelbase)

            ret_dict[id_nr][pos] = (im_p, tcr, thorttle)

    logging.info('Found %d image triplets (L, C, R)', len(ret_dict))

    return ret_dict


if __name__ == "__main__":
    PATH = sys.argv[1]
    PATH_DICT = read(PATH, 20)
