"""For training and saving neural network using Keras.

More info in README.md

"""
# Standard imports
import logging

# Local imports
import DrivingLog
import Utils

# Dependecy imports
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam
import tensorflow as tf

import numpy as np
from sklearn.utils import shuffle

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

        _dataset, _labels = Utils.random_shifts(_dataset, _labels, 22, 16)
        _dataset = Utils.random_brightness(_dataset)
        _dataset = Utils.random_shadows(_dataset)

        start += batch_size
        yield _dataset, _labels


def train_generator_v2(dataset, labels, batch_size):
    """Training generator."""
    dataset_size = labels.shape[0]

    start = 0
    while True:
        end = start + batch_size

        if end <= dataset_size:
            _dataset, _labels = np.copy(dataset[start:end]), np.copy(labels[start:end])
        else:
            diff = end - dataset_size
            _dataset = np.concatenate((np.copy(dataset[start:]), np.copy(dataset[0:diff])), axis=0)
            _labels = np.concatenate((np.copy(labels[start:]), np.copy(labels[0:diff])), axis=0)
            start = 0

            dataset, labels = shuffle(dataset, labels)

        _dataset, _labels = Utils.random_shifts(_dataset, _labels, 22, 16)
        _dataset = Utils.random_brightness(_dataset)
        _dataset = Utils.random_shadows(_dataset)

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
DRIVE_INFO = DrivingLog.read(FLAGS.train_path, 165)
DATASET, LABELS, _ = prep_trainingset(DRIVE_INFO)
TRAING_GEN = train_generator_v2(DATASET, LABELS, FLAGS.batch_size)

# Valid dataset and generator if val_path (validation driving log path) is provided
if FLAGS.val_path != '':
    VAL_DRIVE_INFO = DrivingLog.read(FLAGS.val_path, 165)
    VALID_DATA, VALID_LABELS = prep_dataset(VAL_DRIVE_INFO, False)

SAMPLES, _ = next(TRAING_GEN)  # load random samples necesarry for model creation
KERAS_MODEL = model(SAMPLES)  # creates model

logging.info('Batch Size : %d', FLAGS.batch_size)

if FLAGS.keras_weights != '':
    logging.info('Pretrained weights loaded')
    KERAS_MODEL.load_weights(FLAGS.keras_weights)

try:
    if FLAGS.val_path != '':
        KERAS_MODEL.fit_generator(
            generator=TRAING_GEN,
            steps_per_epoch=FLAGS.batch_size,
            epochs=FLAGS.epochs,
            verbose=1,
            validation_data=(VALID_DATA, VALID_LABELS),
            use_multiprocessing=True,
            workers=12
        )
    else:
        KERAS_MODEL.fit_generator(
            generator=TRAING_GEN,
            steps_per_epoch=FLAGS.batch_size,
            epochs=FLAGS.epochs,
            verbose=1,
            use_multiprocessing=True,
            workers=12
        )

    # If output_model dir path provided save model. #
    if FLAGS.output_model != '':
        Utils.save_keras_model(KERAS_MODEL, FLAGS.output_model)
except KeyboardInterrupt:
    # Early stopping possible by interruping script. If output_model dir path provided save model. #
    if FLAGS.output_model != '':
        Utils.save_keras_model(KERAS_MODEL, FLAGS.output_model)
