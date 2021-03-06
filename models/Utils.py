"""Project helping functions."""

# Standard imports
import logging
import math

# Dependecy imports
import cv2
import numpy as np
from scipy.misc import imresize


def load_data(drive_info, wheelbase=20, use_left_right=True):
    """Load images from files.

    Randomly will load one of three possible images for each path - left, center or right.
    For left and right images there will be steering angle adjusted see shift_ang variables.
    It helps model to better learn steering angles.

    Output is same size as paths.
    """

    dataset = []
    labels = []

    for entry in drive_info:

        if use_left_right:
            i_lrc = np.random.randint(3)
            if i_lrc == 0:
                im_path, steer, thorttle = entry['L']
                shift_ang = 3.6 # Shift by degrees
            if i_lrc == 1:
                im_path, steer, thorttle = entry['C']
                shift_ang = 0.
            if i_lrc == 2:
                im_path, steer, thorttle = entry['R']
                shift_ang = -3.6 # Shift by degrees
        else:
            im_path, steer, thorttle = entry['C']
            shift_ang = 0.

        im_side = cv2.imread(im_path)
        im_side = cv2.cvtColor(im_side, cv2.COLOR_BGR2RGB)

        dataset.append(im_side)

        steer += shift_ang # add shift angle

        tcr = angle2tcr(float(steer), wheelbase)
        labels.append(tcr)

    return np.array(dataset), np.array(labels)

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

def resize_im(images):
    """Resize give image dataset."""
    new_w = 200
    new_h = 66
    res_images = np.zeros((images.shape[0], new_h, new_w, images.shape[3]), dtype=images.dtype)

    for i, img in enumerate(images):
        res_images[i] = imresize(img, (new_h, new_w))

    return res_images

def save_keras_model(save_model, path):
    """Save keras model to given path."""
    save_model.save_weights(path + 'model.h5')

    with open(path + 'model.json', "w") as text_file:
        text_file.write(save_model.to_json())

    logging.info('Keras json model saved. %s', path + 'model.json')
    logging.info('Keras h5 model saved. %s', path + 'model.h5')

def random_brightness(images):
    """Add random brightness to give image dataset to imitate day/night."""

    for i, img in enumerate(images):
        image1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        random_bright = .25 + np.random.uniform()
        image1[:, :, 2] = image1[:, :, 2] * random_bright
        image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

        images[i] = image1

    return images

def random_shifts(images, labels, h_shift, v_shift):
    """Add random horizontal/vertical shifts to image dataset to imitate
    steering away from sides."""

    rows = images[0].shape[0]
    cols = images[0].shape[1]

    for i, img in enumerate(images):
        horizontal = h_shift * np.random.uniform() - h_shift / 2
        vertical = v_shift * np.random.uniform() - v_shift / 2

        warp_m = np.float32([[1, 0, horizontal], [0, 1, vertical]])

        # change also corresponding lable -> steering angle
        labels[i] = labels[i] + horizontal / h_shift * 2 * .2
        images[i] = cv2.warpAffine(img, warp_m, (cols, rows))

    return images, labels

def cut_rows(dataset, top=0, buttom=0):
    """Remove specific rows from top and buttom from given image dataset."""
    return dataset[:, 0 + top:160 - buttom, :, :]

def random_shadows(images):
    """Add random shadows to given image dataset. It helps model to generalize better."""

    for i, img in enumerate(images):
        top_y = 320 * np.random.uniform()
        top_x = 0
        bot_x = 160
        bot_y = 320 * np.random.uniform()
        image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        shadow_mask = 0 * image_hls[:, :, 1]
        x_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][0]
        y_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][1]

        shadow_mask[((x_m - top_x) * (bot_y - top_y) -
                     (bot_x - top_x) * (y_m - top_y) >= 0)] = 1

        if np.random.randint(2) == 1:
            random_bright = .5
            cond1 = shadow_mask == 1
            cond0 = shadow_mask == 0
            if np.random.randint(2) == 1:
                image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
            else:
                image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
        images[i] = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return images
