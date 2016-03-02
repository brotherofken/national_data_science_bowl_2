from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import logging
logging.basicConfig(format = u'[%(asctime)s]  %(message)s', level = logging.INFO)

from model import get_model
from utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation

seed = 1377713

def load_train_data():
    """
    Load training data from .npy files.
    """
    tX = np.load('../input/X_train.npy')
    ty = np.load('../input/y_train.npy')

    tX = tX.astype(np.float32)
    
    np.random.seed(seed)
    shuffle_idx = np.random.permutation(np.arange(tX.shape[0]))
    tX = tX[shuffle_idx]
    ty = ty[shuffle_idx]
    
    vX = np.load('../input/X_valid.npy').astype(np.float32)
    vy = np.load('../input/y_valid.npy')
        
    return tX, ty, vX, vy


def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split]
    X_train = X[split:, :, :, :]
    y_train = y[split:]

    return X_train, y_train, X_test, y_test


def train():
    """
    Training model.
    """
    logging.info('Loading and compiling models...')
    model = get_model()

    logging.info('Loading training data...')
    X_train, y_train, X_test, y_test = load_train_data()

    logging.info('Pre-processing train images...')
    X_train = preprocess(X_train)
    
    logging.info('Pre-processing validation images...')
    X_test = preprocess(X_test)

    # split to training and test
    #X_train, y_train, X_test, y_test = split_data(X, y, split_ratio=0.15)

    metric = 'mse'
    nb_iter = 600
    epochs_per_iter = 1
    batch_size = 32
    calc_crps = 1  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_rmse = sys.float_info.max
    min_val_loss_crps = sys.float_info.max

    logging.info('-'*50)
    logging.info('Training...')
    logging.info('-'*50)

    for i in range(nb_iter):
        logging.info('-'*50)
        logging.info('Iteration {0}/{1}'.format(i + 1, nb_iter))
        logging.info('-'*50)

        logging.info('Augmenting images - rotations')
        X_train_aug = rotation_augmentation(X_train, 20)
        logging.info('Augmenting images - shifts')
        X_train_aug = shift_augmentation(X_train_aug, 0.1, 0.1)

        logging.info('Fitting model...')
        hist = model.fit(X_train_aug, y_train, shuffle=True, nb_epoch=epochs_per_iter,
                                         batch_size=batch_size, validation_data=(X_test, y_test))

        # sigmas for predicted data, actually loss function values (RMSE)
        loss = hist.history['loss'][-1]
        val_loss = hist.history['val_loss'][-1]

        if calc_crps > 0 and i % calc_crps == 0:
            logging.info('Evaluating CRPS...')
            pred = model.predict(X_train, batch_size=batch_size, verbose=1)
            val_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

            # CDF for train and test data (actually a step function)
            cdf_train = real_to_cdf(y_train)
            cdf_test = real_to_cdf(y_test)

            # CDF for predicted data
            cdf_pred = real_to_cdf(pred, loss)
            cdf_val_pred = real_to_cdf(val_pred, val_loss)

            # evaluate CRPS on training data
            crps_train = crps(cdf_train, cdf_pred)
            logging.info('CRPS(train) = {0}'.format(crps_train))

            # evaluate CRPS on test data
            crps_test = crps(cdf_test, cdf_val_pred)
            logging.info('CRPS(test) = {0}'.format(crps_test))

        logging.info('Saving weights...')
        # save weights so they can be loaded later
        model.save_weights('../models/weights/weights_combine.hdf5', overwrite=True)

        # for best (lowest) val losses, save weights
        if metric == 'rmse':
            if val_loss < min_val_loss_rmse:
                min_val_loss_rmse = val_loss
                model.save_weights('../models/weights/weights_combine_rmse_best.hdf5', overwrite=True)
        else:
            if crps_test < min_val_loss_crps:
                min_val_loss_crps = crps_test
                model.save_weights('../models/weights/weights_combine_crps_best.hdf5', overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('./logs/val_loss.txt', mode='w+') as f:
            f.write(str(min_val_loss_rmse))
            f.write('\n')
            f.write(str(min_val_loss_crps))


train()
