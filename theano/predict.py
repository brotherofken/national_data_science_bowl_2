import argparse
import theano
import lasagne
import theano.tensor as T
import numpy as np
from utils import load_network, print_network
IMAGE_SIZE = 90

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
                        "--mean-file",
                        type=str,
                        default='',
                        help="Path to the file storing network configuration")
    parser.add_argument("-d",
                        "--data",
                        type=str,
                        default='',
                        help="Data file")

    args = parser.parse_args()

    #####################
    #  Build the model  #
    #####################
    net = load_network(args.network)
    print("Loaded network: ")

    print_network(net)

    # allocate symbolic variables for theano graph computations
    X_batch = T.tensor4('x')

    data = np.load(args.data)

    if args.mean_file:
        mean = np.load(args.mean_file)

    if args.mean_file:
        data = data - mean
    x_test = np.rollaxis(data, 3, 1)

    # allocate shared variables for images, labels and learing rate
    x_shared = theano.shared(np.zeros((x_test.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE), dtype=theano.config.floatX),
                             borrow=True)

    preds = lasagne.layers.get_output(net['output'], X_batch, deterministic=True)
    #feature_output = lasagne.layers.get_output(net['output'], X_batch, deterministic=True)

    print("Compiling theano functions...")
    # create theano functions for calculating losses on train and validation sets
    iter_valid = theano.function(
        [],
        preds,
        givens={
            X_batch: x_shared #[batch_index * BATCH_SIZE: (batch_index + 1) * BATCH_SIZE],
        },
    )

    ###################
    #   Predictions   #
    ###################
    ppreds = []
    for i in xrange((992 // 64) + 1):
        x_test2 = x_test[i * 64: (i+1) * 64]
        preds = np.zeros((5, x_test2.shape[0]), dtype=theano.config.floatX)

        x_shared.set_value(x_test2[:, :, :90, :90])
        preds[0] = iter_valid().ravel()
        x_shared.set_value(x_test2[:, :, 10:, 10:])
        preds[1] = iter_valid().ravel()

        x_shared.set_value(x_test2[:, :, 10:, :90])
        preds[2] = iter_valid().ravel()

        x_shared.set_value(x_test2[:, :, :90, 10:])
        preds[3] = iter_valid().ravel()

        x_shared.set_value(x_test2[:, :, 5:95, 5:95])
        out = iter_valid().ravel()
        preds[4] = out
        
        ppreds.append(np.mean(preds, axis=0))

    res = np.concatenate(ppreds)
    print res.shape
    np.save("preds", res)

    # elapsed_time = time.time() - now
    # print("The best auc: %.5f obtained on epoch %d.\n The training took %d seconds." %
    #       (best_auc, best_epoch, elapsed_time))
    # print(" The average performance was %.1f images/sec" % (
    #     (len(x_train) + len(y_train)) * float(epoch) / elapsed_time))
    #
    # results = np.array([train_loss, valid_loss, aucs], dtype=np.float)
    # np.save("data/tidy/training.npy", results)
    # transform.terminate()
    # transform.join()


