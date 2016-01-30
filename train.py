import argparse
import logging
import os
import time
from multiprocessing import Queue
import lasagne
from lasagne.regularization import regularize_network_params, l2
from transform import Transformer
from utils import *


def train(net, train_set, valid_set, train_params, logger=None, prefix=''):
    # unpack arguments
    x_train, y_train = train_set
    x_valid, y_valid = valid_set
    # convert to bc01 order, train set will be converted in Transformer
    #x_valid = np.rollaxis(x_valid, 3, 1)
    x_valid = x_valid[:, np.newaxis, ...]

    BATCH_SIZE = train_params['batch_size']
    IMAGE_SIZE = train_params['image_size']
    MOMENTUM = train_params['momentum']
    MAX_EPOCH = train_params['epochs']
    LEARNING_RATE_SCHEDULE = train_params['lr_schedule']
    L2 = train_params.get('L2', 0.)
    output = net['output']

    print("Starting dataset loader...")
    queue = Queue(5)
    transform = Transformer(x_train, y_train, queue, batch_size=BATCH_SIZE)
    transform.start()



    # allocate symbolic variables for theano graph computations
    batch_index = T.iscalar('batch_index')
    X_batch = T.tensor4('x')
    y_batch = T.fmatrix('y')

    # allocate shared variables for images, labels and learing rate
    x_shared = theano.shared(np.zeros((BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=theano.config.floatX),
                             borrow=True)
    y_shared = theano.shared(np.zeros((BATCH_SIZE, 2), dtype=theano.config.floatX),
                             borrow=True)

    learning_rate = theano.shared(np.float32(LEARNING_RATE_SCHEDULE[0]))

    out_train = lasagne.layers.get_output(output, X_batch, deterministic=False)
    out_val = lasagne.layers.get_output(output, X_batch, deterministic=True)

    loss_train = T.mean(lasagne.objectives.squared_error(out_train, y_batch))# + L2 * regularize_network_params(output, l2)
    loss_val = T.mean(lasagne.objectives.squared_error(out_val, y_batch))# + L2 * regularize_network_params(output, l2)

    # collect all model parameters
    all_params = lasagne.layers.get_all_params(output)
    # generate parameter updates for SGD with Nesterov momentum
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, MOMENTUM)

    logger.info("Compiling theano functions...")
    # create theano functions for calculating losses on train and validation sets
    iter_train = theano.function(
        [],
        [loss_train],
        updates=updates,
        givens={
            X_batch: x_shared, #[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE],
            y_batch: y_shared, #[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE],
            },
        )
    iter_valid = theano.function(
        [],
        [loss_val, out_val],
        givens={
            X_batch: x_shared,
            y_batch: y_shared,
            },
        )

    ###################
    # Actual training #
    ###################

    n_train_batches = x_train.shape[0] // BATCH_SIZE
    n_val_batches = x_valid.shape[0] // BATCH_SIZE
    # keep track of networks best performance and save net configuration
    best_epoch = 0
    best_valid = 1.
    best_auc = 0.
    # epoch and iteration counters
    epoch = 0
    _iter = 0
    # wait for at least this many epochs before saving the model
    min_epochs = 0
    # store these values for learning curves plotting
    train_loss = []
    valid_loss = []
    aucs = []

    # wait for this many epochs if the validation error is not increasing
    patience = 10
    now = time.time()
    logger.info("| Epoch | Train err | Validation err | ROC AUC | Ratio |  Time  |")
    logger.info("|---------------------------------------------------------------|")

    try:
        # get next chunks of data
        while epoch < MAX_EPOCH:
            if epoch in LEARNING_RATE_SCHEDULE:
                learning_rate.set_value(LEARNING_RATE_SCHEDULE[epoch])
            epoch += 1
            x_next, y_next = queue.get()

            losses = []
            while x_next is not None:
                
                x_shared.set_value(x_next, borrow=True)
                y_shared.set_value(y_next, borrow=True)
                l = iter_train()
                losses.append(l)
                x_next, y_next = queue.get()

            avg_train_loss = np.mean(losses)

            # average the predictions across 5 patches: corners and center
            losses = []
            for idx in xrange(n_val_batches - 1):
                x_shared.set_value(x_valid[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE])
                y_shared.set_value(y_valid[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE])
                vloss, out_val = iter_valid()
                losses.append(vloss)

            avg_valid_loss = np.mean(losses)

            logger.info("|%6d | %9.6f | %14.6f | %7.5f | %1.3f | %6d |" %
                        (epoch,
                         avg_train_loss,
                         avg_valid_loss,
                         0,
                         avg_valid_loss / avg_train_loss,
                         time.time() - now))
            # keep track of these for future analysis
            train_loss.append(avg_train_loss)
            valid_loss.append(avg_valid_loss)

            # if this is the best kappa obtained so far
            # save the model to make predictions on the test set
            # if auc > best_auc:
            #     # always wait for min_epochs, to avoid frequent saving
            #     # during early stages of learning
            #     if epoch >= min_epochs:
            #         save_network(net, filename=os.path.join(prefix, 'net.pickle'))
            #         np.save(os.path.join(prefix, "val_predictions.npy"), valid_probas)
            #         valid_features = feats / 5
            #         np.save(os.path.join(prefix, "val_features.npy"), valid_features)
            #     best_auc = auc
            #     best_epoch = epoch
            #     patience = 10
    except KeyboardInterrupt:
        logger.info("Trainig interrupted on epoch %d" % epoch)

    elapsed_time = time.time() - now
    logger.info("The best auc: %.5f obtained on epoch %d.\n The training took %d seconds." %
          (best_auc, best_epoch, elapsed_time))
    logger.info(" The average performance was %.1f images/sec" % (
        (len(x_train) + len(y_train)) * float(epoch) / elapsed_time))

    results = np.array([train_loss, valid_loss, aucs], dtype=np.float)
    np.save(os.path.join(prefix, "training.npy"), results)
    transform.terminate()
    transform.join()

if __name__ == '__main__':
    #####################
    # Get cmd arguments #
    #####################
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",
                        "--network",
                        type=str,
                        help="Path to the pickled network file")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        default='',
                        help="Path to the file storing network configuration")
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        help="Number of epochs to train the network")
    args = parser.parse_args()

    #####################
    #  Build the model  #
    #####################
    # if args.model:
    #     execfile(args.model)
    #     output = net['output']
    #     print("Built model:")
    # elif args.network:
    #     all_layers, output = load_network(args.network)
    #     print("Loaded network: ")

    net, train_params = load_config(args.model)

    print_network(net)
    
    mean_img = np.load("data/mean_96.npy")
    train_set = np.load("data/timages_96.npy") - mean_img, np.load("data/tlabels_96.npy").astype(np.float32)
    valid_set = np.load("data/vimages_96.npy") - mean_img, np.load("data/vlabels_96.npy").astype(np.float32)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    prefix = 'data/tidy/%s' % args.model
    if not os.path.isdir(prefix):
        os.makedirs(prefix)

    train(net, train_set, valid_set, train_params, logger=logger, prefix=prefix)
