"""Training script, this is converted from a ipython notebook
"""

import os
import csv
import sys
import numpy as np
import mxnet as mx
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format = u'[%(asctime)s]  %(message)s', level = logging.INFO)

# In[2]:

def get_lenet():
    """ A lenet style net, takes difference of each frame as input.
    """
    source = mx.sym.Variable("data")
    source = (source - 128) * (1.0/128)
    frames = mx.sym.SliceChannel(source, num_outputs=30)
    diffs = [frames[i+1] - frames[i] for i in range(29)]
    source = mx.sym.Concat(*diffs)
    net = mx.sym.Convolution(source, kernel=(5, 5), num_filter=40)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=40)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(net)
    flatten = mx.symbol.Dropout(flatten)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=600)
    # Name the final layer as softmax so it auto matches the naming of data iterator
    # Otherwise we can also change the provide_data in the data iter
    return mx.symbol.LogisticRegressionOutput(data=fc1, name='softmax')

def get_symbol():
    source = mx.sym.Variable("data")
    source = (source - 128) * (1.0/128)
    frames = mx.sym.SliceChannel(source, num_outputs=30)
    diffs = [frames[i+1] - frames[i] for i in range(29)]
    source = mx.sym.Concat(*diffs)
    # group 1
    conv1_1 = mx.symbol.Convolution(data=source, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = mx.symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = mx.symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    pool3 = mx.symbol.Pooling(
        data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    pool4 = mx.symbol.Pooling(
        data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    pool5 = mx.symbol.Pooling(
        data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=2048, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=2048, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # output
    fc8 = mx.symbol.FullyConnected(data=drop7, num_hidden=600, name="fc8")
    return mx.symbol.LogisticRegressionOutput(data=fc8, name='softmax')


    # model.add(Convolution2D(64, 3, 3, border_mode='same'))
    # model.add(LeakyReLU(0.2))
    # model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    # model.add(LeakyReLU(0.2))
    # model.add(ZeroPadding2D(padding=(1, 1)))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Convolution2D(96, 3, 3, border_mode='same'))
    # model.add(LeakyReLU(0.2))
    # model.add(Convolution2D(96, 3, 3, border_mode='valid'))
    # model.add(LeakyReLU(0.2))
    # model.add(ZeroPadding2D(padding=(1, 1)))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Convolution2D(128, 2, 2, border_mode='same'))
    # model.add(LeakyReLU(0.2))
    # model.add(Convolution2D(128, 2, 2, border_mode='valid'))
    # model.add(LeakyReLU(0.2))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Flatten())
    # model.add(Dense(1024, W_regularizer=l2(1e-3)))
    # model.add(LeakyReLU(0.2))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))




def get_alexnet():
    source = mx.sym.Variable("data")
    source = (source - 128) * (1.0/128)
    frames = mx.sym.SliceChannel(source, num_outputs=30)
    diffs = [frames[i+1] - frames[i] for i in range(29)]
    source = mx.sym.Concat(*diffs)
    
    conv1 = mx.symbol.Convolution(
        data=source, kernel=(3, 3), pad = (1, 1), num_filter=64)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    conv2 = mx.symbol.Convolution(
        data=relu1, kernel=(3, 3), pad = (1, 1), num_filter=64)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu2, pool_type = "max", kernel=(2, 2), stride=(2, 2))
    dropout1 = mx.symbol.Dropout(data=pool1, p=0.25)

    conv3 = mx.symbol.Convolution(
        data=dropout1, kernel=(3, 3), pad = (1, 1), num_filter=96)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 3), pad = (1, 1), num_filter=96)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu4, pool_type = "max", kernel=(2, 2), stride=(2, 2))
    dropout2 = mx.symbol.Dropout(data=pool2, p=0.25)

    conv5 = mx.symbol.Convolution(
        data=dropout2, kernel=(3, 3), pad = (1, 1), num_filter=128)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    conv6 = mx.symbol.Convolution(
        data=relu5, kernel=(3, 3), pad = (1, 1), num_filter=128)
    relu6 = mx.symbol.Activation(data=conv6, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu6, pool_type = "max", kernel=(2, 2), stride=(2, 2))
    dropout3 = mx.symbol.Dropout(data=pool3, p=0.25)

    flatten = mx.symbol.Flatten(data=dropout3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=1024)
    relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
    dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
    # stage 6
    fc3 = mx.symbol.FullyConnected(data=dropout1, num_hidden=600)

    return mx.symbol.LogisticRegressionOutput(data=fc3, name='softmax')


def CRPS(label, pred):
    """ Custom evaluation metric on CRPS.
    """
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1] - 1):
            if pred[i, j] > pred[i, j + 1]:
                pred[i, j + 1] = pred[i, j]
    return np.sum(np.square(label - pred)) / label.size


# In[3]:

def encode_label(label_data):
    """Run encoding to encode the label into the CDF target.
    """
    systole = label_data[:, 1]
    diastole = label_data[:, 2]
    systole_encode = np.array([
            (x < np.arange(600)) for x in systole
        ], dtype=np.uint8)
    diastole_encode = np.array([
            (x < np.arange(600)) for x in diastole
        ], dtype=np.uint8)
    return systole_encode, diastole_encode

def encode_csv(label_csv, systole_csv, diastole_csv):
    systole_encode, diastole_encode = encode_label(np.loadtxt(label_csv, delimiter=","))
    np.savetxt(systole_csv, systole_encode, delimiter=",", fmt="%g")
    np.savetxt(diastole_csv, diastole_encode, delimiter=",", fmt="%g")

# Write encoded label into the target csv
# We use CSV so that not all data need to sit into memory
# You can also use inmemory numpy array if your machine is large enough
encode_csv("../input/train-label.csv", "../input/train-systole.csv", "../input/train-diastole.csv")


# # Training the systole net

# In[4]:

network = get_alexnet()
batch_size = 32
devs = [mx.gpu(0)]
data_train = mx.io.CSVIter(data_csv="../input/train-128x128-data.csv", data_shape=(30, 128, 128),
                           label_csv="../input/train-systole.csv", label_shape=(600,),
                           batch_size=batch_size)

data_validate = mx.io.CSVIter(data_csv="../input/validate-128x128-data.csv", data_shape=(30, 128, 128),
                              batch_size=1)

systole_model = mx.model.FeedForward(ctx=devs,
        symbol             = network,
        num_epoch          = 10,
        learning_rate      = 0.1,
        wd                 = 0.0001,
        momentum           = 0.9)

systole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))

prefix = 'systole_model'
iteration = 10
systole_model.save(prefix, iteration)

# load model back
systole_model = mx.model.FeedForward.load(prefix, iteration, ctx=devs, num_epoch = 150, learning_rate = 0.01)
systole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))

prefix = 'systole_model'
iteration = 150
systole_model.save(prefix, iteration)

# load model back
systole_model = mx.model.FeedForward.load(prefix, iteration, ctx=devs, num_epoch = 200, learning_rate = 0.001)
systole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))

prefix = 'systole_model'
iteration = 200
systole_model.save(prefix, iteration)


# # Predict systole

# In[5]:

systole_prob = systole_model.predict(data_validate)


# # Training the diastole net

# In[6]:

network = get_alexnet()
batch_size = 32
devs = [mx.gpu(0)]
data_train = mx.io.CSVIter(data_csv="../input/train-128x128-data.csv", data_shape=(30, 128, 128),
                           label_csv="../input/train-diastole.csv", label_shape=(600,),
                           batch_size=batch_size)

diastole_model = mx.model.FeedForward(ctx=devs,
        symbol             = network,
        num_epoch          = 10,
        learning_rate      = 0.1,
        wd                 = 0.0001,
        momentum           = 0.9)

diastole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))

prefix = 'diastole_model'
iteration = 10
diastole_model.save(prefix, iteration)

# load model back
diastole_model = mx.model.FeedForward.load(prefix, iteration, ctx=devs, num_epoch = 150, learning_rate = 0.01)
diastole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))

prefix = 'diastole_model'
iteration = 150
diastole_model.save(prefix, iteration)

# load model back
diastole_model = mx.model.FeedForward.load(prefix, iteration, ctx=devs, num_epoch = 200, learning_rate = 0.001)
diastole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))

prefix = 'diastole_model'
iteration = 200
diastole_model.save(prefix, iteration)


# # Predict diastole

# In[7]:

diastole_prob = diastole_model.predict(data_validate)


# # Generate Submission

# In[8]:

def accumulate_result(validate_lst, prob):
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    fi = csv.reader(open(validate_lst))
    for i in range(size):
        line = fi.__next__() # Python2: line = fi.next()
        idx = int(line[0])
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]))
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result


# In[9]:

systole_result = accumulate_result("../input/validate-label.csv", systole_prob)
diastole_result = accumulate_result("../input/validate-label.csv", diastole_prob)


# In[10]:

# we have 2 person missing due to frame selection, use udibr's hist result instead
def doHist(data):
    h = np.zeros(600)
    for j in np.ceil(data).astype(int):
        h[j:] += 1
    h /= len(data)
    return h
train_csv = np.genfromtxt("../input/train-label.csv", delimiter=',')
hSystole = doHist(train_csv[:, 1])
hDiastole = doHist(train_csv[:, 2])


# In[11]:

def submission_helper(pred):
    p = np.zeros(600)
    pred.resize(p.shape)
    p[0] = pred[0]
    for j in range(1, 600):
        a = p[j - 1]
        b = pred[j]
        if b < a:
            p[j] = a
        else:
            p[j] = b
    return p



# In[12]:

fi = csv.reader(open("../input/sample_submission_validate.csv"))
f = open("../submissions/submission_mxnet.csv", "w")
fo = csv.writer(f, lineterminator='\n')
fo.writerow(fi.__next__())
for line in fi:
    idx = line[0]
    key, target = idx.split('_')
    key = int(key)
    out = [idx]
    if key in systole_result:
        if target == 'Diastole':
            out.extend(list(submission_helper(diastole_result[key])))
        else:
            out.extend(list(submission_helper(systole_result[key])))
    else:
        print("Miss: %s" % idx)
        if target == 'Diastole':
            out.extend(hDiastole)
        else:
            out.extend(hSystole)
    fo.writerow(out)
f.close()
