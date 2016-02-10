import lasagne
from lasagne import layers
from lasagne.nonlinearities import very_leaky_rectify, sigmoid
from lasagne.layers import dnn
from collections import OrderedDict


train_params = {
    'image_size': 96,
    'batch_size': 128,
    'momentum': 0.9,
    'epochs': 500,
    'lr_schedule': {
        0: 0.001,
        100: 0.0001,
        200: 0.00001,
    }
}

def build_model():
    net = OrderedDict()
    net['input'] = layers.InputLayer(shape=(train_params['batch_size'],
                                            1,
                                            train_params['image_size'],
                                            train_params['image_size']))

    net['conv11'] = dnn.Conv2DDNNLayer(net['input'],
                                       num_filters=64,
                                       filter_size=(3, 3),
                                       W=lasagne.init.Orthogonal(gain='relu'),
                                       pad='same',
                                       nonlinearity=very_leaky_rectify)
    net['conv12'] = dnn.Conv2DDNNLayer(net['conv11'],
                                       num_filters=64,
                                       filter_size=(3, 3),
                                       pad='same',
                                       W=lasagne.init.Orthogonal(gain='relu'),
                                       nonlinearity=very_leaky_rectify)
    net['pool1'] = dnn.MaxPool2DDNNLayer(net['conv12'], (3, 3), stride=(2, 2))

    net['conv21'] = dnn.Conv2DDNNLayer(net['pool1'],
                                       num_filters=96,
                                       filter_size=(3, 3),
                                       pad='same',
                                       W=lasagne.init.Orthogonal(gain='relu'),
                                       nonlinearity=very_leaky_rectify)
    net['conv22'] = dnn.Conv2DDNNLayer(net['conv21'],
                                       num_filters=96,
                                       filter_size=(3, 3),
                                       pad='same',
                                       W=lasagne.init.Orthogonal(gain='relu'),
                                       nonlinearity=very_leaky_rectify)
    net['pool2'] = dnn.MaxPool2DDNNLayer(net['conv22'], (3, 3), stride=(2, 2))

    net['conv31'] = dnn.Conv2DDNNLayer(net['pool2'],
                                       num_filters=128,
                                       filter_size=(3, 3),
                                       pad='same',
                                       W=lasagne.init.Orthogonal(gain='relu'),
                                       nonlinearity=very_leaky_rectify)
    net['conv32'] = dnn.Conv2DDNNLayer(net['conv31'],
                                       num_filters=128,
                                       filter_size=(3, 3),
                                       pad='same',
                                       W=lasagne.init.Orthogonal(gain='relu'),
                                       nonlinearity=very_leaky_rectify)
    net['conv33'] = dnn.Conv2DDNNLayer(net['conv32'],
                                       num_filters=128,
                                       filter_size=(3, 3),
                                       pad='same',
                                       W=lasagne.init.Orthogonal(gain='relu'),
                                       nonlinearity=very_leaky_rectify)
    net['pool3'] = dnn.MaxPool2DDNNLayer(net['conv33'], (3, 3), stride=(2, 2))

    net['conv41'] = dnn.Conv2DDNNLayer(net['pool3'],
                                       num_filters=196,
                                       filter_size=(3, 3),
                                       pad='same',
                                       W=lasagne.init.Orthogonal(gain='relu'),
                                       nonlinearity=very_leaky_rectify)
    net['conv42'] = dnn.Conv2DDNNLayer(net['conv41'],
                                       num_filters=196,
                                       filter_size=(3, 3),
                                       pad='same',
                                       W=lasagne.init.Orthogonal(gain='relu'),
                                       nonlinearity=very_leaky_rectify)
    net['conv43'] = dnn.Conv2DDNNLayer(net['conv42'],
                                       num_filters=256,
                                       filter_size=(3, 3),
                                       pad='same',
                                       W=lasagne.init.Orthogonal(gain='relu'),
                                       nonlinearity=very_leaky_rectify)
    net['pool4'] = dnn.MaxPool2DDNNLayer(net['conv43'], (3, 3), stride=(2, 2))
    #net['drop0'] = lasagne.layers.DropoutLayer(net['pool4'], p=0.5)

    net['fc1'] = layers.DenseLayer(net['pool4'],
                                   num_units=256,
                                   W=lasagne.init.Normal(),
                                   nonlinearity=None)
    net['maxout1'] = layers.FeaturePoolLayer(net['fc1'], 2)
    net['drop1'] = lasagne.layers.DropoutLayer(net['maxout1'], p=0.5)

    net['fc2'] = layers.DenseLayer(net['drop1'],
                                   num_units=256,
                                   W=lasagne.init.Normal(),
                                   nonlinearity=None)
    net['maxout2'] = layers.FeaturePoolLayer(net['fc2'], 2)
    net['drop2'] = lasagne.layers.DropoutLayer(net['maxout2'], p=0.5)

    net['output'] = layers.DenseLayer(net['drop2'],
                                      num_units=2,
                                      nonlinearity=None)
    return net
