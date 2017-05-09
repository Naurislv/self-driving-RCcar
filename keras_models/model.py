"""For training and saving neural network using Keras.

More info in README.md

"""
import driving_log

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam

import tensorflow as tf

import cv2
import numpy as np
from scipy.misc import imresize

import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


class train_path_error(Exception):
    """Custom Training log error,- happens when argument is not passed."""

    pass


flags = tf.app.flags
FLAGS = flags.FLAGS

# Input arguments

flags.DEFINE_string('train_path', '', "Training images directory.")
flags.DEFINE_string('val_path', '', "Validation images directory.")
flags.DEFINE_string('output_model', '', "Directory to save keras model. e.g. /home/user/udacity/")
flags.DEFINE_string('keras_weights', '', "Pretrained wieghts to use for training. Keras h5 file.")
flags.DEFINE_integer('epochs', 10, "Training epoch count.")
flags.DEFINE_integer('batch_size', 256, "Batch size")
flags.DEFINE_integer('dropout', 0, "0 - Disable | 1 - Enable dropout.")

if FLAGS.train_path == '':
    raise train_path_error('Please provide argument : --train_path path_to_training_images')


class Utils(object):
    """Helping functions."""

    def load_data(self, drive_info, Path, use_left_right=True):
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
                if (i_lrc == 0):
                    im_path, steer, thorttle = drive_info[key]['L']
                    shift_ang = .18
                if (i_lrc == 1):
                    im_path, steer, thorttle = drive_info[key]['C']
                    shift_ang = 0.
                if (i_lrc == 2):
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

        for i in range(len(images)):
            res_images[i] = imresize(images[i], (new_h, new_w))

        return res_images

    def save_keras_model(self, model, path):
        """Save keras model to given path."""
        model.save_weights(path + 'model.h5')

        with open(path + 'model.json', "w") as text_file:
            text_file.write(m.to_json())

        logging.info('\n\nKeras model saved.')

    def random_brightness(self, images):
        """Add random brightness to give image dataset to imitate day/night."""

        for i in range(len(images)):
            image1 = cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV)
            random_bright = .25 + np.random.uniform()
            image1[:, :, 2] = image1[:, :, 2] * random_bright
            image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

            images[i] = image1

        return images

    def random_shifts(self, images, labels, h_shift, v_shift):
        """Add random horizontal/vertical shifts to image dataset to imitate steering away from sides."""

        rows = images[0].shape[0]
        cols = images[0].shape[1]

        for i in range(len(images)):
            horizontal = h_shift * np.random.uniform() - h_shift / 2
            vertical = v_shift * np.random.uniform() - v_shift / 2

            M = np.float32([[1, 0, horizontal], [0, 1, vertical]])

            # change also corresponding lable -> steering angle
            labels[i] = labels[i] + horizontal / h_shift * 2 * .2
            images[i] = cv2.warpAffine(images[i], M, (cols, rows))

        return images, labels

    def cut_rows(self, dataset, up=0, down=0):
        """Remove specific rows from up and down from given image dataset."""
        return dataset[:, 0 + up:160 - down, :, :]

    def random_shadows(self, images):
        """Add random shadows to given image dataset. It helps model to generalize better."""

        for i in range(len(images)):
            top_y = 320 * np.random.uniform()
            top_x = 0
            bot_x = 160
            bot_y = 320 * np.random.uniform()
            image_hls = cv2.cvtColor(images[i], cv2.COLOR_RGB2HLS)
            shadow_mask = 0 * image_hls[:, :, 1]
            X_m = np.mgrid[0:images[i].shape[0], 0:images[i].shape[1]][0]
            Y_m = np.mgrid[0:images[i].shape[0], 0:images[i].shape[1]][1]

            shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1

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


utils = Utils()
# Main Functions #


def prep_dataset(drive_info, path, use_left_right=True):
    """Prepare dataset for keras model."""
    dataset, labels = utils.load_data(drive_info, path, use_left_right)

    return dataset, labels


def prep_trainingset(drive_info, path):
    """Prepare dataset for training."""
    dataset, labels = prep_dataset(drive_info, path)
    labels = labels.astype(np.float16)

    logging.info('Training dataset shape {}'.format(dataset.shape))
    dataset_size = labels.shape[0]

    return dataset, labels, dataset_size


def train_generator(dataset, labels, batch_size):
    """Training generator."""
    dataset_size = labels.shape[0]

    start = 0
    while True:
        end = start + batch_size

        d = np.copy(dataset)
        l = np.copy(labels)

        if end <= dataset_size:
            d, l = d[start:end], l[start:end]
        else:
            diff = end - dataset_size
            d = np.concatenate((d[start:], d[0:diff]), axis=0)
            l = np.concatenate((l[start:], l[0:diff]), axis=0)
            start = 0

        d, l = utils.random_shifts(d, l, 22, 16)
        d = utils.random_brightness(d)
        d = utils.random_shadows(d)

        start += batch_size
        yield d, l


def model(data):
    """Create keras model.

    Based on : https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    """

    dshape = data.shape

    model = Sequential()
    model.add(BatchNormalization(input_shape=(dshape[1], dshape[2], dshape[3])))
    model.add(Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    if FLAGS.dropout == 1:
        model.add(Dropout(0.4))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    if FLAGS.dropout == 1:
        model.add(Dropout(0.4))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=adam(lr=0.0001))

    return model


logging.info('Preparing dataset/-s, may take awhile')

# Training dataset and generator #
drive_info = driving_log.read(FLAGS.train_path, 20)
dataset, labels, dataset_size = prep_trainingset(drive_info, FLAGS.train_path)
tg = train_generator(dataset, labels, FLAGS.batch_size)

# Valid dataset and generator if val_path (validation driving log path) is provided #
if FLAGS.val_path != '':
    val_drive_info = driving_log.read(FLAGS.val_path, 20)
    valid_data, valid_lables = prep_dataset(val_drive_info, FLAGS.val_path, False)

samples, _ = next(tg)  # load random samples necesarry for model creation
m = model(samples)  # creates model

logging.info('Batch Size : {}'.format(FLAGS.batch_size))

if FLAGS.keras_weights != '':
    logging.info('Pretrained weights loaded')
    m.load_weights(FLAGS.keras_weights)

try:
    if FLAGS.val_path != '':
        m.fit_generator(generator=tg, steps_per_epoch=FLAGS.batch_size, epochs=FLAGS.epochs, verbose=1,
                        validation_data=(valid_data, valid_lables), pickle_safe=True, workers=12)
    else:
        m.fit_generator(generator=tg, steps_per_epoch=FLAGS.batch_size, epochs=FLAGS.epochs, verbose=1,
                        pickle_safe=True, workers=12)

    # If output_model dir path provided save model. #
    if FLAGS.output_model != '':
        utils.save_keras_model(m, FLAGS.output_model)
except KeyboardInterrupt:
    # Early stopping possible by interruping script. If output_model dir path provided save model. #
    if FLAGS.output_model != '':
        utils.save_keras_model(m, FLAGS.output_model)
