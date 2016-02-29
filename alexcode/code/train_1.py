from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import logging
logging.basicConfig(format = u'[%(asctime)s]  %(message)s', level = logging.INFO)

from model import get_model
from utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation


def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load('../input/X_train.npy')
    y = np.load('../input/y_train.npy')

    X = X.astype(np.float32)
    X /= 255

    seed = np.random.randint(1, 10e6)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X, y


def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]

    return X_train, y_train, X_test, y_test


def train():
    """
    Training systole and diastole models.
    """
    logging.info('Loading and compiling models...')
    model_systole = get_model()
    model_diastole = get_model()

    logging.info('Loading training data...')
    X, y = load_train_data()

    logging.info('Pre-processing images...')
    X = preprocess(X)

    # split to training and test
    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.15)

    nb_iter = 300
    epochs_per_iter = 1
    batch_size = 32
    calc_crps = 1  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_crps_test_systole = sys.float_info.max
    min_crps_test_diastole = sys.float_info.max

    logging.info('-'*50)
    logging.info('Training...')
    logging.info('-'*50)

    for i in range(nb_iter):
        logging.info('-'*50)
        logging.info('Iteration {0}/{1}'.format(i + 1, nb_iter))
        logging.info('-'*50)

        logging.info('Augmenting images - rotations')
        X_train_aug = rotation_augmentation(X_train, 30)
        logging.info('Augmenting images - shifts')
        X_train_aug = shift_augmentation(X_train_aug, 0.15, 0.15)

        logging.info('Fitting systole model...')
        hist_systole = model_systole.fit(X_train_aug, y_train[:, 0], shuffle=True, nb_epoch=epochs_per_iter,
                                         batch_size=batch_size, validation_data=(X_test, y_test[:, 0]))

        logging.info('Fitting diastole model...')
        hist_diastole = model_diastole.fit(X_train_aug, y_train[:, 1], shuffle=True, nb_epoch=epochs_per_iter,
                                           batch_size=batch_size, validation_data=(X_test, y_test[:, 1]))

        # sigmas for predicted data, actually loss function values (RMSE)
        loss_systole = hist_systole.history['loss'][-1]
        loss_diastole = hist_diastole.history['loss'][-1]
        val_loss_systole = hist_systole.history['val_loss'][-1]
        val_loss_diastole = hist_diastole.history['val_loss'][-1]

        if calc_crps > 0 and i % calc_crps == 0:
            logging.info('Evaluating CRPS...')
            pred_systole = model_systole.predict(X_train, batch_size=batch_size, verbose=1)
            pred_diastole = model_diastole.predict(X_train, batch_size=batch_size, verbose=1)
            val_pred_systole = model_systole.predict(X_test, batch_size=batch_size, verbose=1)
            val_pred_diastole = model_diastole.predict(X_test, batch_size=batch_size, verbose=1)

            # CDF for train and test data (actually a step function)
            cdf_train_systole = real_to_cdf(y_train[:, 0])
            cdf_train_diastole = real_to_cdf(y_train[:, 1])
            cdf_test_systole = real_to_cdf(y_test[:, 0])
            cdf_test_diastole = real_to_cdf(y_test[:, 1])

            # CDF for predicted data
            cdf_pred_systole = real_to_cdf(pred_systole, loss_systole)
            cdf_pred_diastole = real_to_cdf(pred_diastole, loss_diastole)
            cdf_val_pred_systole = real_to_cdf(val_pred_systole, val_loss_systole)
            cdf_val_pred_diastole = real_to_cdf(val_pred_diastole, val_loss_diastole)

            # evaluate CRPS on training data
            crps_train_systole = crps(cdf_train_systole, cdf_pred_systole)
            logging.info('CRPS(train_systole) = {0}'.format(crps_train_systole))

            crps_train_diastole = crps(cdf_train_diastole, cdf_pred_diastole)
            logging.info('CRPS(train_diastole) = {0}'.format(crps_train_diastole))

            # evaluate CRPS on test data
            crps_test_systole = crps(cdf_test_systole, cdf_val_pred_systole)
            logging.info('CRPS(test_systole) = {0}'.format(crps_test_systole))

            # evaluate CRPS on test data
            crps_test_diastole = crps(cdf_test_diastole, cdf_val_pred_diastole)
            logging.info('CRPS(test_diastole) = {0}'.format(crps_test_diastole))

            # evaluate CRPS on test data
            crps_test = crps(real_to_cdf(np.concatenate((y_test[:, 0], y_test[:, 1]))), np.concatenate((cdf_val_pred_systole, cdf_val_pred_diastole)))
            logging.info('CRPS(test) = {0}'.format(crps_test))

        logging.info('Saving weights...')
        # save weights so they can be loaded later
        model_systole.save_weights('../models/weights/weights_systole_1.hdf5', overwrite=True)
        model_diastole.save_weights('../models/weights/weights_diastole_1.hdf5', overwrite=True)

        # for best (lowest) val losses, save weights
        if crps_test_systole < min_crps_test_systole:
            min_crps_test_systole = crps_test_systole
            min_val_loss_systole = val_loss_systole
            model_systole.save_weights('../models/weights/weights_systole_best_1.hdf5', overwrite=True)

        if crps_test_diastole < min_crps_test_diastole:
            min_crps_test_diastole = crps_test_diastole
            min_val_loss_diastole = val_loss_diastole
            model_diastole.save_weights('../models/weights/weights_diastole_best_1.hdf5', overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('./logs/val_loss_1.txt', mode='w+') as f:
            f.write(str(min_val_loss_systole))
            f.write('\n')
            f.write(str(min_val_loss_diastole))
            f.write('\n')
            f.write(str(min_crps_test_systole))
            f.write('\n')
            f.write(str(min_crps_test_diastole))


train()
