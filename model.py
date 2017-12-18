#coding=utf-8
import tensorflow as tf
import numpy as np

DEFAULT_PADDING = 'SAME'
WEIGHT_DECAY_FACTOR = 0.

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           initializer=tf.contrib.layers.xavier_initializer())
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _conv(name, in_, ksize, strides=[1,1,1,1], padding=DEFAULT_PADDING):
    """create convolution layer 
    Args:
      name: name of the variable
      in_: input nodes
      ksize: kernel size for convolute
      strides: offset between each kernel
      padding: padding format for image's overflow
    Returns:
      Variable Tensor
    """    
    n_kern = ksize[3]
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=ksize, wd=WEIGHT_DECAY_FACTOR)
        conv = tf.nn.conv2d(in_, kernel, strides, padding=padding)
        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope.name)

    print name, conv.get_shape().as_list()
    return conv

def _maxpool(name, in_, ksize, strides, padding=DEFAULT_PADDING):
    """create max-pooling layer
    Args:
      name: name of the variable
      in_: input nodes
      ksize: kernel size for pooling
      strides: offset between each kernel
      padding: padding format for image's overflow
    Returns:
      Variable Tensor
    """  
    pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides,
                          padding=padding, name=name)

    print name, pool.get_shape().as_list()
    return pool

def _fc(name, in_, outsize, dropout=1.0):
    """create fully-connected layer
    Args:
      name: name of the variable
      in_: input nodes
      outsize: dimension of output nodes
      dropout: dropout rate to invalid nodes
    Returns:
      Variable Tensor
    """  
    with tf.variable_scope(name) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        insize = in_.get_shape().as_list()[-1]
        weights = _variable_with_weight_decay('weights', shape=[insize, outsize], wd=0.004)
        biases = _variable_on_cpu('biases', [outsize], tf.constant_initializer(0.0))
        fc = tf.nn.relu(tf.matmul(in_, weights) + biases, name=scope.name)
        fc = tf.nn.dropout(fc, dropout)

    print name, fc.get_shape().as_list()
    return fc

def inference(img, keep_prob, n_classes):
    """Foward layer of AlexModel including 5 conv layers, 3 fc layers"""

    conv1 = _conv('conv1', img, [11, 11, 3, 96], [1, 4, 4, 1], 'VALID')
    pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2 = _conv('conv2', pool1, [5, 5, 96, 256])
    pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    conv3 = _conv('conv3', pool2, [3, 3, 256, 384])

    conv4 = _conv('conv4', conv3, [3, 3, 384, 384])

    conv5 = _conv('conv5', conv4, [3, 3, 384, 256])
    pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')


    # transform to single vector for fc
    shape = pool5.get_shape().as_list()  
    pool5_vector = tf.reshape(pool5, [-1, np.prod(shape[1:])])

    fc6 = _fc('fc6', pool5_vector, 4096, dropout=keep_prob)

    fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob)

    fc8 = _fc('fc8', fc7, n_classes)

    return fc8

def loss(fc8, labels):
    """simply cross entropy (remove weight decay loss for simplicity)"""
    l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fc8)
    # add weight decay
    #l = tf.reduce_mean(l)
    #tf.add_to_collection('losses', l)
    return tf.reduce_mean(l)

def classify(fc8):
    """softmax of last layer and output index of highest probability"""
    softmax = tf.nn.softmax(fc8)
    y = tf.argmax(softmax, 1)
    return y

def load_alexnet(sess, caffetf_modelpath):
    """ load pre-trained alex model from  alexnet_imagenet.npy"""

    def load(name, layer_data, group=1):
        w, b = layer_data

        if group != 1:
            w = np.concatenate((w, w), axis=2) 

        with tf.variable_scope(name, reuse=True):
            for subkey, data in zip(('weights', 'biases'), (w, b)):
                print 'loading ', name, subkey
                var = tf.get_variable(subkey)
                sess.run(var.assign(data))

    caffemodel = np.load(caffetf_modelpath)
    data_dict = caffemodel.item()
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
        name = l
        # historical grouping by alexnet
        if l == 'conv2' or l == 'conv4' or l == 'conv5':
            load(name, data_dict[l], group=2)
        else:
            load(name, data_dict[l])

    load('fc6', data_dict['fc6'])
    load('fc7', data_dict['fc7'])
