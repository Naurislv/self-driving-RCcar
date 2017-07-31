"""For training and saving neural network using Keras.

More info in README.md

"""
# Standard imports
import logging
import math

# Local imports
import driving_log

# ML imports
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam
import tensorflow as tf

# Other imports
import cv2
import numpy as np
from scipy.misc import imresize

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


_FLAGS = tf.app.flags
FLAGS = _FLAGS.FLAGS

# Input arguments
_FLAGS.DEFINE_string('train_path', '', "Training images directory.")
_FLAGS.DEFINE_string('val_path', '', "Validation images directory.")
_FLAGS.DEFINE_string('output_model', '', "Directory to save keras model. e.g. /home/user/udacity/")
_FLAGS.DEFINE_string('keras_weights', '', "Pretrained wieghts to use for training. Keras h5 file.")
_FLAGS.DEFINE_integer('epochs', 10, "Training epoch count.")
_FLAGS.DEFINE_integer('batch_size', 256, "Batch size")
_FLAGS.DEFINE_integer('dropout', 0, "0 - Disable | 1 - Enable dropout.")

if FLAGS.train_path == '':
    raise OSError('Please provide argument : --train_path path_to_training_images')


class Utils(object):
    """Helping functions."""

    def load_data(self, drive_info, use_left_right=True):
        """Load images from files.

        Randomly will load one of three possible images for each path - left, center or right.
        For left and right images there will be steering angle adjusted see shift_ang variables.
        It helps model to better learn steering angles.

        Output is same size as paths.
        """

        dataset = []
        labels = []

        for key in drive_info:

            if use_left_right:
                i_lrc = np.random.randint(3)
                if i_lrc == 0:
                    im_path, steer, thorttle = drive_info[key]['L']
                    shift_ang = .18
                if i_lrc == 1:
                    im_path, steer, thorttle = drive_info[key]['C']
                    shift_ang = 0.
                if i_lrc == 2:
                    im_path, steer, thorttle = drive_info[key]['R']
                    shift_ang = -.18
            else:
                im_path, steer, thorttle = drive_info[key]['C']
                shift_ang = 0.

            im_side = cv2.imread(im_path)
            im_side = cv2.cvtColor(im_side, cv2.COLOR_BGR2RGB)

            dataset.append(im_side)
            labels.append(steer + shift_ang)

        return np.array(dataset), np.array(labels)

    def resize_im(self, images):
        """Resize give image dataset."""
        new_w = 200
        new_h = 66
        res_images = np.zeros((images.shape[0], new_h, new_w, images.shape[3]), dtype=images.dtype)

        for i, img in enumerate(images):
            res_images[i] = imresize(img, (new_h, new_w))

        return res_images

    def save_keras_model(self, save_model, path):
        """Save keras model to given path."""
        save_model.save_weights(path + 'model.h5')

        with open(path + 'model.json', "w") as text_file:
            text_file.write(save_model.to_json())

        logging.info('\n\nKeras model saved.')

    def random_brightness(self, images):
        """Add random brightness to give image dataset to imitate day/night."""

        for i, img in enumerate(images):
            image1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            random_bright = .25 + np.random.uniform()
            image1[:, :, 2] = image1[:, :, 2] * random_bright
            image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

            images[i] = image1

        return images

    def random_shifts(self, images, labels, h_shift, v_shift):
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

    def cut_rows(self, dataset, top=0, buttom=0):
        """Remove specific rows from top and buttom from given image dataset."""
        return dataset[:, 0 + top:160 - buttom, :, :]

    def random_shadows(self, images):
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


UTILS = Utils()


def prep_dataset(drive_info, use_left_right=True):
    """Prepare dataset for keras model."""
    dataset, labels = UTILS.load_data(drive_info, use_left_right)

    return dataset, labels


def prep_trainingset(drive_info):
    """Prepare dataset for training."""
    dataset, labels = prep_dataset(drive_info)
    labels = labels.astype(np.float16)

    logging.info('Training dataset shape %s', dataset.shape)
    dataset_size = labels.shape[0]

    return dataset, labels, dataset_size


def train_generator(dataset, labels, batch_size):
    """Training generator."""
    dataset_size = labels.shape[0]

    start = 0
    while True:
        end = start + batch_size

        _dataset = np.copy(dataset)
        _labels = np.copy(labels)

        if end <= dataset_size:
            _dataset, _labels = _dataset[start:end], _labels[start:end]
        else:
            diff = end - dataset_size
            _dataset = np.concatenate((_dataset[start:], _dataset[0:diff]), axis=0)
            _labels = np.concatenate((_labels[start:], _labels[0:diff]), axis=0)
            start = 0

        _dataset, _labels = UTILS.random_shifts(_dataset, _labels, 22, 16)
        _dataset = UTILS.random_brightness(_dataset)
        _dataset = UTILS.random_shadows(_dataset)

        start += batch_size
        yield _dataset, _labels


def model(data):
    """Create keras model.

    Based on : https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    """

    dshape = data.shape

    seq_model = Sequential()
    seq_model.add(BatchNormalization(input_shape=(dshape[1], dshape[2], dshape[3])))
    seq_model.add(Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    seq_model.add(Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    seq_model.add(Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    seq_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    seq_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    seq_model.add(Flatten())
    seq_model.add(Dense(100, activation='relu'))
    if FLAGS.dropout == 1:
        seq_model.add(Dropout(0.4))
    seq_model.add(Dense(50, activation='relu'))
    seq_model.add(Dense(10, activation='relu'))
    if FLAGS.dropout == 1:
        seq_model.add(Dropout(0.4))
    seq_model.add(Dense(1))

    seq_model.compile(loss='mse', optimizer=adam(lr=0.0001))

    return seq_model


logging.info('Preparing dataset/-s, may take awhile')

# Training dataset and generator
# 165 mm wheelbasae for wltoys a969
# https://www.banggood.com/Wltoys-A969-Rc-Car-118-2_4Gh-4WD-Short-Course-Truck-p-916962.html
DRIVE_INFO = driving_log.read(FLAGS.train_path, 165)
DATASET, LABELS, _ = prep_trainingset(DRIVE_INFO)
TRAING_GEN = train_generator(DATASET, LABELS, FLAGS.batch_size)

# Valid dataset and generator if val_path (validation driving log path) is provided
if FLAGS.val_path != '':
    VAL_DRIVE_INFO = driving_log.read(FLAGS.val_path, 165)
    VALID_DATA, VALID_LABELS = prep_dataset(VAL_DRIVE_INFO, False)

SAMPLES, _ = next(TRAING_GEN)  # load random samples necesarry for model creation
KERAS_MODEL = model(SAMPLES)  # creates model

logging.info('Batch Size : %d', FLAGS.batch_size)

if FLAGS.keras_weights != '':
    logging.info('Pretrained weights loaded')
    KERAS_MODEL.load_weights(FLAGS.keras_weights)

try:
    if FLAGS.val_path != '':
        KERAS_MODEL.fit_generator(generator=TRAING_GEN, steps_per_epoch=FLAGS.batch_size,
                                  epochs=FLAGS.epochs, verbose=1,
                                  validation_data=(VALID_DATA, VALID_LABELS),
                                  pickle_safe=True, workers=12)
    else:
        KERAS_MODEL.fit_generator(generator=TRAING_GEN, steps_per_epoch=FLAGS.batch_size,
                                  epochs=FLAGS.epochs, verbose=1, pickle_safe=True, workers=12)

    # If output_model dir path provided save model. #
    if FLAGS.output_model != '':
        UTILS.save_keras_model(KERAS_MODEL, FLAGS.output_model)
except KeyboardInterrupt:
    # Early stopping possible by interruping script. If output_model dir path provided save model. #
    if FLAGS.output_model != '':
        UTILS.save_keras_model(KERAS_MODEL, FLAGS.output_model)
