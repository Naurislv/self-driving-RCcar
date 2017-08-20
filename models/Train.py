"""For training and saving neural network using Keras.

More info in README.md

"""
# Standard imports
import logging
import os
import random

# Local imports
import DrivingLog
import Utils

# Dependecy imports
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam
from keras.utils.data_utils import Sequence
import keras.backend.tensorflow_backend as K

import tensorflow as tf

import numpy as np

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# Remove Tensorflow logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '6'

# Set Keras TF backend allow_growth not to consume all GPU memory
K_CONFIG = K.tf.ConfigProto()
K_CONFIG.allow_soft_placement = True
K_CONFIG.gpu_options.allow_growth = True # pylint: disable=E1101
K.set_session(K.tf.Session(config=K_CONFIG))

_FLAGS = tf.app.flags
FLAGS = _FLAGS.FLAGS

# Input arguments
_FLAGS.DEFINE_string('train_path', '', "Training images directory.")
_FLAGS.DEFINE_string('output_model', '', "Directory to save keras model. e.g. /home/user/udacity/")
_FLAGS.DEFINE_string('keras_weights', '', "Pretrained wieghts to use for training. Keras h5 file.")
_FLAGS.DEFINE_integer('epochs', 10, "Training epoch count.")
_FLAGS.DEFINE_integer('batch_size', 256, "Batch size")
_FLAGS.DEFINE_integer('dropout', 0, "0 - Disable | 1 - Enable dropout.")
_FLAGS.DEFINE_integer('batch_norm', 0, "0 - Enable batch normalization for inputs.")
_FLAGS.DEFINE_string('activation', 'selu', "SELU - Set all activation functions to this.")

if FLAGS.train_path == '':
    raise OSError('Please provide argument : --train_path path_to_training_images')

def rand_batch_idx(dataset_size, batch_size, idx):
    rand_idx = random.sample(range(dataset_size), dataset_size)

    idx = 0
    while True:
        if idx == 0:
            rand_idx = random.sample(range(dataset_size), dataset_size)

        yield rand_idx[idx * batch_size:(idx + 1) * batch_size]

class DatasetSequence(Sequence):
    """Keras utils Sequence implementation for fit_generator

    Docs : https://keras.io/utils/
    """

    def __init__(self, driving_log, batch_size):
        """Everything defined here can't be change from other functions."""
        self.driving_log = np.array(driving_log)
        self.batch_size = batch_size

        logging.warning("DataSequence Implementation is not complete. "
                        "See __getitem__ and fix if you can")

    def __len__(self):
        return len(self.driving_log) // self.batch_size

    def __getitem__(self, idx):

        # Very bad workaround for shuffling data. This is because I cant figure out how to
        # modify class variable
        rand_idx = random.sample(range(self.batch_size), self.batch_size)

        batch_x, batch_y = Utils.load_data(
            # self.driving_log[idx * self.batch_size:(idx + 1) * self.batch_size]
            self.driving_log[rand_idx],
            wheelbase=165,
            use_left_right=True
        )

        batch_x, batch_y = Utils.random_shifts(batch_x, batch_y, 22, 16)
        batch_x = Utils.random_brightness(batch_x)
        batch_x = Utils.random_shadows(batch_x)

        return batch_x, batch_y

def model(dshape):
    """Create keras model.

    Based on : https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    """
    seq_model = Sequential()

    if FLAGS.batch_norm == 1:
        seq_model.add(BatchNormalization(input_shape=(dshape[0], dshape[1], dshape[2])))
        seq_model.add(Conv2D(24, (5, 5), padding='valid',
                             activation=FLAGS.activation, strides=(2, 2)))
    else:
        seq_model.add(Conv2D(24, (5, 5), padding='valid', activation=FLAGS.activation,
                             strides=(2, 2), input_shape=(dshape[0], dshape[1], dshape[2])))
    seq_model.add(Conv2D(36, (5, 5), padding='valid', activation=FLAGS.activation, strides=(2, 2)))
    seq_model.add(Conv2D(48, (5, 5), padding='valid', activation=FLAGS.activation, strides=(2, 2)))
    seq_model.add(Conv2D(64, (3, 3), padding='valid', activation=FLAGS.activation))
    seq_model.add(Conv2D(64, (3, 3), padding='valid', activation=FLAGS.activation))
    seq_model.add(Flatten())
    seq_model.add(Dense(100, activation=FLAGS.activation))
    if FLAGS.dropout == 1:
        seq_model.add(Dropout(0.4))
    seq_model.add(Dense(50, activation=FLAGS.activation))
    if FLAGS.dropout == 1:
        seq_model.add(Dropout(0.3))
    seq_model.add(Dense(10, activation=FLAGS.activation))
    if FLAGS.dropout == 1:
        seq_model.add(Dropout(0.1))
    seq_model.add(Dense(1))

    seq_model.compile(loss='mse', optimizer=adam(lr=0.0001), metrics=['mse'])

    # seq_model.summary()

    return seq_model


logging.info('Preparing dataset/-s, may take awhile')

# Training dataset and generator
# 165 mm wheelbasae for wltoys a969
# https://www.banggood.com/Wltoys-A969-Rc-Car-118-2_4Gh-4WD-Short-Course-Truck-p-916962.html
DRIVE_INFO = DrivingLog.read(FLAGS.train_path)
DATASET_SIZE = len(DRIVE_INFO)

TRAIN_GEN = DatasetSequence(DRIVE_INFO, FLAGS.batch_size)

KERAS_MODEL = model((110, 200, 3))  # creates model

logging.info('Batch Size : %d', FLAGS.batch_size)

if FLAGS.keras_weights != '':
    logging.info('Pretrained weights loaded')
    KERAS_MODEL.load_weights(FLAGS.keras_weights)

try:
    HISTORY = KERAS_MODEL.fit_generator(
        generator=TRAIN_GEN,
        steps_per_epoch=DATASET_SIZE / FLAGS.batch_size,
        epochs=FLAGS.epochs,
        verbose=1,
        use_multiprocessing=True,
        workers=16
    )

    # If output_model dir path provided save model. #
    if FLAGS.output_model != '':
        SAVE_PATH = "{}_E[{}]_BS[{}]_DO[{}]_A[{}]_BN[{}]_L[{}]_".format(
            FLAGS.output_model,
            FLAGS.epochs,
            FLAGS.batch_size,
            FLAGS.dropout,
            FLAGS.activation,
            FLAGS.batch_norm,
            HISTORY.history['loss'][-1]  # Get last loss entry
        )

        Utils.save_keras_model(KERAS_MODEL, SAVE_PATH)
except KeyboardInterrupt:
    # Early stopping possible by interruping script. If output_model dir path provided save model. #
    if FLAGS.output_model != '':
        Utils.save_keras_model(KERAS_MODEL, FLAGS.output_model)
