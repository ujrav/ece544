"""
models.py: Definitions for generator and discriminator convnets.
"""

import tensorflow as tf


# commonly used stride settings for 2D convolutions
STRIDE_1 = [1, 1, 1, 1]
STRIDE_2 = [1, 2, 2, 1]


#Manju: commented the following function for batch normalization that this original guy actually wrote manually.
#I did so because the new tensorflow has a built-in function that does better than the following hard coded one. 


# def _bn(bottom, is_train):
#     """
#         Creates a batch normalization op.
#         Meant to be invoked from other layer operations.
#     """

#     # assume inference by default
    

#     # create scale and shift variables
#     bn_shape = bottom.get_shape()[-1]
#     shift = tf.get_variable("beta", shape=bn_shape,
#                             initializer=tf.constant_initializer(0.0))
#     scale = tf.get_variable("gamma", shape=bn_shape,
#                             initializer=tf.constant_initializer(1.0))

#     # compute mean and variance
#     bn_axes = list(range(len(bottom.get_shape()) - 1))
#     (mu, var) = tf.nn.moments(bottom, bn_axes)

#     # batch normalization ops
#     with tf.variable_scope(tf.get_variable_scope(), reuse=False):
#         ema = tf.train.ExponentialMovingAverage(decay=0.99)

#     def train_op():
#         with tf.variable_scope(tf.get_variable_scope(), reuse=False):
#             ema_op = ema.apply([mu, var])
#         with tf.control_dependencies([ema_op]):
#             return (tf.identity(mu), tf.identity(var))

#     def test_op():
#         moving_mu = ema.average(mu)
#         moving_var = ema.average(var)
#         return (moving_mu, moving_var)

#     (mean, variance) = tf.cond(is_train, train_op, test_op)

#     top = tf.nn.batch_normalization(bottom, mean, variance, shift, scale, 1e-4)

#     return top


def conv2d(name, bottom, shape, strides, top_shape=None, with_bn=True, is_train=None):
    """
        Creates a convolution + BN block.
    """
    

    with tf.variable_scope(name) as scope:
        
        # apply batch normalization, if necessary
        # Manju: The new batch normalization function is built-in:
        bn=tf.cond(is_train,
                   lambda: tf.contrib.layers.batch_norm(bottom, decay=0.99,  epsilon=0.0001, is_training=is_train, updates_collections=None,reuse=None, scope=scope) if with_bn else bottom, 
                   lambda: tf.contrib.layers.batch_norm(bottom, decay=0.99, epsilon=0.0001, is_training=is_train, updates_collections=None,reuse=True, scope=scope) if with_bn else bottom)

        # add convolution op
        weights = tf.get_variable("weights", shape=shape,
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        if top_shape is not None:
            conv = tf.nn.conv2d_transpose(bn, weights, top_shape, strides, padding="SAME")
        else:
            conv = tf.nn.conv2d(bn, weights, strides, padding="SAME")

        # add biases
        bias_shape = [shape[-1]] if top_shape is None else [shape[-2]]
        biases = tf.get_variable("biases", shape=bias_shape,
                                 initializer=tf.constant_initializer())
        top = tf.nn.bias_add(conv, biases)

    return top


def linear(name, bottom, shape, with_bn=True, is_train=None):
    """
        Creates a fully connected + BN block.
    """
    

    with tf.variable_scope(name) as scope:

        bn=tf.cond(is_train,
                   lambda: tf.contrib.layers.batch_norm(bottom, decay=0.99, epsilon=0.0001, is_training=is_train, updates_collections=None,reuse=None, scope=scope) if with_bn else bottom, 
                   lambda: tf.contrib.layers.batch_norm(bottom, decay=0.99, epsilon=0.0001, is_training=is_train, updates_collections=None,reuse=True, scope=scope) if with_bn else bottom)

        # inner product
        weights = tf.get_variable("weights", shape=shape,
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        linear = tf.matmul(bn, weights)

        # add biases
        bias_shape = [shape[-1]]
        biases = tf.get_variable("biases", shape=bias_shape,
                                 initializer=tf.constant_initializer())
        top = tf.nn.bias_add(linear, biases)

    return top

def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def generator(data, is_train, side_length):
    """
        Builds the generator model.

        :param tensorflow.Tensor data:
            A 4-D input tensor with the last dimension Gaussian sampled.

        :param tensorflow.Tensor is_train:
            Determines whether or not the generator is used only for inference.

        :param int side_length:
            Image side length.
    """

    

    assert side_length % 16 == 0, "image side length must be divisible by 16"

    dim = side_length / 16
    (batch_size, z_len) = data.get_shape().as_list()

    # project block, 1024 outputs
    proj = tf.nn.relu(linear("g_proj", data, [z_len, dim * dim * 1024], with_bn=False,is_train=is_train))
    rshp = tf.reshape(proj, [batch_size, dim, dim, 1024])

    # conv1 block, 512 outputs
    dim *= 2
    conv1 = tf.nn.relu(conv2d("g_conv1", rshp, [5, 5, 512, 1024], STRIDE_2,
                              top_shape=[batch_size, dim, dim, 512], is_train=is_train))

    # conv2 block, 256 outputs
    dim *= 2
    conv2 = tf.nn.relu(conv2d("g_conv2", conv1, [5, 5, 256, 512], STRIDE_2,
                              top_shape=[batch_size, dim, dim, 256], is_train=is_train))

    # conv3 block, 128 outputs
    dim *= 2
    conv3 = tf.nn.relu(conv2d("g_conv3", conv2, [5, 5, 128, 256], STRIDE_2,
                              top_shape=[batch_size, dim, dim, 128], is_train=is_train))

    # conv4 block, 64 outputs
    dim *= 2
    conv4 = tf.nn.relu(conv2d("g_conv4", conv3, [5, 5, 64, 128], STRIDE_2,
                              top_shape=[batch_size, dim, dim, 64], is_train=is_train))

    # conv5 block, 3 outputs
    conv5 = tf.nn.tanh(conv2d("g_conv5", conv4, [3, 3, 3, 64], STRIDE_1,
                              top_shape=[batch_size, dim, dim, 3], is_train=is_train))

    top = conv5

    return top


def discriminator(data, is_train,reuse=False):
    """
        Builds the discriminator network.

        :param tensorflow.Tensor data:
            A 4-D tensor representing an input image.

        :param tensorflow.Tensor is_train:
            Determines whether or not the discriminator is used only for inference.
    """

    # conv1 block, 128 outputs
    conv1 = lrelu(conv2d("d_conv1", data, [5, 5, 3, 128], STRIDE_2, with_bn=False, is_train=is_train))

    # conv2 block, 256 outputs
    conv2 = lrelu(conv2d("d_conv2", conv1, [5, 5, 128, 256], STRIDE_2, is_train=is_train))

    # conv3 block, 512 outputs
    conv3 = lrelu(conv2d("d_conv3", conv2, [5, 5, 256, 512], STRIDE_2, is_train=is_train))

    # conv4 block, 1024 outputs
    conv4 = lrelu(conv2d("d_conv4", conv3, [5, 5, 512, 1024], STRIDE_2, is_train=is_train))

    # average pooling: (Manju) We NEED this.
    avg_pool = tf.reduce_mean(conv4, [1, 2])

    # fully connected
    classifier = linear("d_classifier", avg_pool, [1024, 1], with_bn=False,is_train=is_train) 

    top = classifier

    return top

def discriminator_features(data, is_train,reuse=False):
    """
        Builds the discriminator network.

        :param tensorflow.Tensor data:
            A 4-D tensor representing an input image.

        :param tensorflow.Tensor is_train:
            Determines whether or not the discriminator is used only for inference.
    """

    # conv1 block, 128 outputs
    conv1 = lrelu(conv2d("d_conv1", data, [5, 5, 3, 128], STRIDE_2, with_bn=False, is_train=is_train))

    # conv2 block, 256 outputs
    conv2 = lrelu(conv2d("d_conv2", conv1, [5, 5, 128, 256], STRIDE_2, is_train=is_train))

    # conv3 block, 512 outputs
    conv3 = lrelu(conv2d("d_conv3", conv2, [5, 5, 256, 512], STRIDE_2, is_train=is_train))

    # conv4 block, 1024 outputs
    conv4 = lrelu(conv2d("d_conv4", conv3, [5, 5, 512, 1024], STRIDE_2, is_train=is_train))

    # average pooling: (Manju) We NEED this.
    avg_pool = tf.reduce_mean(conv4, [1, 2])

    top = avg_pool

    return top

def discriminator_classify(data, n_classes, is_train, feature_size = 1024, layer_name="c_classifier_n"):

    features = discriminator_features(data, is_train = tf.constant(False), reuse = True)

    classifier = linear(layer_name, features, [feature_size, n_classes], with_bn = False, is_train=is_train)

    return classifier

def discriminator_classify_to_labels(classification_scores):

    output_classes = tf.where(classification_scores >= 0, tf.ones_like(classification_scores), -1*tf.ones_like(classification_scores))

    return output_classes

def discriminator_classify_exclusive(data, n_classes, is_train, feature_size = 1024, layer_name="c_classifier_n"):

    features = discriminator_features(data, is_train = tf.constant(False), reuse = True)

    classifier = linear(layer_name, features, [feature_size, n_classes], with_bn = False, is_train=is_train)

    return classifier

def discriminator_classify_to_labels_exclusive(classification_scores):

    output_classes = tf.argmax(classification_scores, axis = 1)

    return output_classes