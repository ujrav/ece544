import argparse
import os
import random

# library imports
import numpy as np
from skimage.color import gray2rgb
# from skimage.io import imread, imsave
from skimage.transform import resize
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import scipy.misc
import numpy as np
import math
import cPickle

# local imports
from models_cifar import *

import matplotlib.pyplot as pl


import pdb


# argparse
parser = argparse.ArgumentParser(description="Use a trained DCGAN Descriminator for Classification")
parser.add_argument("-c", "--checkpoint-dir", type=str, default="saved_checkpoints/cifar_dcgan/", help="directory to checkpoints")
parser.add_argument("-t", "--train-dir", type=str, default="data/cifar_10", help="path to training data")
parser.add_argument("-i", "--image-size", type=int, default=64, help="(square) image size")
parser.add_argument("-s", "--scale-size", type=int, default=64, help="resize length for center crop")
parser.add_argument("-e", "--num-epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-o", "--output-dir", type=str, default="output_disc_class/", help="directory for outputs")
parser.add_argument("-m", "--train-mode", type=str, default = "top")
parser.add_argument("-d", "--restore-dir", type=str, default="output_disc_class/", help="directory to restore classification model")
parser.add_argument("-r", "--restore", action="store_true", help="specify to use the latest checkpoint")
parser.add_argument("-z", "--job-dir", type=str, default="outputs", help="blah")


def _read_and_preprocess(paths, scale_len, crop_len):
    """
        Reads multiple images (and labels).
    """

    imgs = []

    for path in paths:
        
        file=file_io.FileIO(path,mode='r')
        img = scipy.misc.imread(file,mode='RGB')

        # force 3-channel images
        # if img.ndim == 2:
        #     img = gray2rgb(img)
        # elif img.shape[2] == 4:
        #     img = img[:, :, :3]

        # compute the resize dimension
        resize_f = float(scale_len) / min(img.shape[:2])
        new_dims = (int(np.round(img.shape[0] * resize_f)),
                    int(np.round(img.shape[1] * resize_f)))

        # prevent the input image from blowing up
        # factor of 2 is more or less an arbitrary number
        max_dim = 2 * scale_len
        new_dims = (min(new_dims[0], max_dim),
                    min(new_dims[1], max_dim))

        # resize and center crop
        img = resize(img, new_dims)
        top = int(np.ceil((img.shape[0] - crop_len) / 2.0))
        left = int(np.ceil((img.shape[1] - crop_len) / 2.0))
        img = img[top:(top+crop_len), left:(left+crop_len)]

        # preprocessing (tanh)
        img = img*2 - 1

        imgs.append(img)

    return np.array(imgs)

def _read_preprocess_cifar10(cifar_path, cifar_filenames):
    

    cifar_data = None

    cifar_labels = None

    for filename in cifar_filenames:

        filepath = os.path.join(cifar_path, filename)

        file = file_io.FileIO(filepath,mode='r')

        data_dict = cPickle.load(file)

        img_set = data_dict["data"]

        img_set = np.reshape(img_set, (img_set.shape[0],3,32,32))

        img_set = np.swapaxes(np.swapaxes(img_set, 1, 2), 2, 3)

        label_set = data_dict["labels"]

        if cifar_data is None:
            cifar_data = img_set
        else:
            cifar_data = np.concatenate((cifar_data, img_set))

        if cifar_labels is None:
            cifar_labels = label_set
        else:
            cifar_labels = np.concatenate((cifar_labels, label_set))

    cifar_data = cifar_data.astype(np.float32, copy=False)/255*2 - 1

    return cifar_data, cifar_labels

def classify_cifar(args):


    scale_len = args.scale_size
    crop_len = args.image_size
    checkpoint_dir = args.checkpoint_dir

    batch_size = 100
    num_epochs = args.num_epochs
    n_classes = 10
    # which attribute columns will we classify
    class_cols = [1,2,3]
    train_path = args.train_dir
    output_path = args.output_dir
    model_path = os.path.join(output_path,"checkpoint/disc_class.tfmodel")

    img_data = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="img_data")
    label_tensor = tf.placeholder(tf.int32, shape = [None,], name="label_tensor")
    is_train = tf.placeholder(tf.bool, name="is_train")

    global_step = tf.Variable(0, name='global_step', trainable=False)


    classifier = discriminator_classify_exclusive(img_data, n_classes, is_train, layer_name="c_classifier_n")
    classifier_labels = discriminator_classify_to_labels_exclusive(classifier)

    classification_loss = tf.reduce_mean(discriminator_classify_exclusive_loss(classifier, label_tensor))

    d_vars = [var for var in tf.trainable_variables() if "d_" in var.name]
    c_vars = [var for var in tf.trainable_variables() if "c_" in var.name]
    train_vars = [var for var in tf.trainable_variables()]

    # saver for discriminator vars only
    saver = tf.train.Saver(var_list = d_vars)

    sess = tf.Session()

    summary_writer = tf.summary.FileWriter('logs', sess.graph)


    # intialize tf variables
    sess.run(tf.global_variables_initializer())

    chkpt_fname = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, chkpt_fname)

    #test
    cifar_batch_files = ["test_batch"]
    cifar_data, cifar_labels = _read_preprocess_cifar10(train_path, cifar_batch_files)
    # cifar_batch_files = ["data_batch_1"]
    # cifar_data, cifar_labels = _read_preprocess_cifar10(train_path, cifar_batch_files)

    with tf.variable_scope("d_conv1") as scope:
        scope.reuse_variables()
        # conv1_filter = tf.transpose(tf.get_variable("weights"), [3, 0, 1, 2])
        # filter_summary = tf.summary.image("layer1_filter", conv1_filter)


        conv1_filter_tf = tf.get_variable("weights")
        conv1_filter = sess.run(conv1_filter_tf)

    with tf.variable_scope("d_conv2") as scope:
        scope.reuse_variables()
        # conv1_filter = tf.transpose(tf.get_variable("weights"), [3, 0, 1, 2])
        # filter_summary = tf.summary.image("layer1_filter", conv1_filter)


        conv2_filter_tf = tf.get_variable("weights")
        conv2_filter = sess.run(conv2_filter_tf)

    with tf.variable_scope("d_conv3") as scope:
        scope.reuse_variables()
        # conv1_filter = tf.transpose(tf.get_variable("weights"), [3, 0, 1, 2])
        # filter_summary = tf.summary.image("layer1_filter", conv1_filter)


        conv3_filter_tf = tf.get_variable("weights")
        conv3_filter = sess.run(conv3_filter_tf)

    with tf.variable_scope("d_conv4") as scope:
        scope.reuse_variables()
        # conv1_filter = tf.transpose(tf.get_variable("weights"), [3, 0, 1, 2])
        # filter_summary = tf.summary.image("layer1_filter", conv1_filter)


        conv4_filter_tf = tf.get_variable("weights")
        conv4_filter = sess.run(conv4_filter_tf)


    for i in range(0,10):

        imgRaw = conv1_filter[:,:,0,i]
        img1 = scipy.misc.imresize(imgRaw, (64,64), interp="nearest")
        pl.imsave("filter_imgs/layer1_"+str(i)+".png", img1)

        imgRaw = conv1_filter[:,:,0,i]
        img2 = scipy.misc.imresize(imgRaw, (64,64), interp="nearest")
        pl.imsave("filter_imgs/layer2_"+str(i)+".png", img2)

        imgRaw = conv1_filter[:,:,0,i]
        img3 = scipy.misc.imresize(imgRaw, (64,64), interp="nearest")
        pl.imsave("filter_imgs/layer3_"+str(i)+".png", img3)

        imgRaw = conv1_filter[:,:,0,i]
        img4 = scipy.misc.imresize(imgRaw, (64,64), interp="nearest")
        pl.imsave("filter_imgs/layer4_"+str(i)+".png", img4)

    # merged = tf.summary.merge_all()
    # summary = sess.run(merged)
    # summary_writer.add_summary(summary)

def main(args):
    #classify_celebA(args)
    classify_cifar(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
