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
from models import *


import pdb


# argparse
parser = argparse.ArgumentParser(description="Use a trained DCGAN Descriminator for Classification")
parser.add_argument("-c", "--checkpoint-dir", type=str, default="output_cifar/checkpoint", help="directory to checkpoints")
parser.add_argument("-i", "--image-size", type=int, default=64, help="(square) image size")
parser.add_argument("-s", "--scale-size", type=int, default=64, help="resize length for center crop")

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

        filepath = cifar_path + filename

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

def classify_celebA(args):


    scale_len = args.scale_size
    crop_len = args.image_size
    checkpoint_dir = args.checkpoint_dir

    batch_size = 100
    n_classes = 3
    # which attribute columns will we classify
    class_cols = [1,2,3]
    train_ratio = 0.8
    img_set = 5000
    dataset_path = "data/celebA/"

    img_data = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="img_data")
    label_tensor = tf.placeholder(tf.float32, shape = [None, n_classes], name="label_tensor")
    is_train = tf.placeholder(tf.bool, name="is_train")

    classifier = discriminator_classify(img_data, n_classes, is_train, layer_name="c_classifier_n")
    classifier_labels = discriminator_classify_to_labels(classifier)

    classification_loss = tf.reduce_mean(tf.reduce_sum((classifier - label_tensor)**2, axis=1))

    d_vars = [var for var in tf.trainable_variables() if "d_" in var.name]
    c_vars = [var for var in tf.trainable_variables() if "c_" in var.name]

    print d_vars
    print "______"
    print c_vars

    opt_classifier = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(classification_loss, var_list=c_vars)

    #read labels
    label_data = np.genfromtxt("data/list_attr_celeba.txt",skip_header=2, dtype=None, usecols=class_cols)
    img_name_data = np.genfromtxt("data/list_attr_celeba.txt",skip_header=2, dtype=None, usecols=(0))

    example_idx = np.arange(0, img_set)

    example_idx_rnd = np.random.permutation(example_idx)

    train_idx = example_idx[0:(int(img_set*0.8))]

    test_idx  = example_idx[(int(img_set*0.8)):]

    num_iter = int(math.ceil(len(train_idx)/batch_size))



    # saver for discriminator vars only
    saver = tf.train.Saver(var_list = d_vars)

    sess = tf.Session()

    # intialize tf variables
    sess.run(tf.global_variables_initializer())

    chkpt_fname = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, chkpt_fname)

    train_idx_shuffle = np.random.permutation(train_idx)

    #train
    for i in range(0, num_iter):
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size

        batch_idx = train_idx_shuffle[batch_start:batch_end]

        batch_labels = label_data[batch_idx, :]

        batch_img_names = img_name_data[batch_idx]

        img_paths = []
        for img_name in batch_img_names:
            img_paths.append(dataset_path + img_name)

        batch_imgs = _read_and_preprocess(img_paths, scale_len, crop_len)

        feed_dict_in = {img_data:batch_imgs, label_tensor:batch_labels, is_train:True}

        _, loss = sess.run([opt_classifier, classification_loss], feed_dict = feed_dict_in)

        print loss

    #test
    test_labels = label_data[test_idx,:]

    test_img_names = img_name_data[test_idx]

    img_paths = []
    for img_name in test_img_names:
            img_paths.append(dataset_path + img_name)

    test_imgs = _read_and_preprocess(img_paths, scale_len, crop_len)

    feed_dict_test = {img_data:test_imgs, label_tensor:test_labels, is_train:True}

    test_loss, test_label_out = sess.run([classification_loss, classifier_labels], feed_dict = feed_dict_test)

    test_acc = np.mean(test_labels == test_label_out, axis = 0)

    print test_acc

    pdb.set_trace()

def classify_cifar(args):


    scale_len = args.scale_size
    crop_len = args.image_size
    checkpoint_dir = args.checkpoint_dir

    batch_size = 100
    num_epochs = 10
    n_classes = 10
    # which attribute columns will we classify
    class_cols = [1,2,3]
    train_path = "data/cifar_10/"

    img_data = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="img_data")
    label_tensor = tf.placeholder(tf.int32, shape = [None,], name="label_tensor")
    is_train = tf.placeholder(tf.bool, name="is_train")

    classifier = discriminator_classify_exclusive(img_data, n_classes, is_train, layer_name="c_classifier_n")
    classifier_labels = discriminator_classify_to_labels_exclusive(classifier)

    classification_loss = tf.reduce_mean(discriminator_classify_exclusive_loss(classifier, label_tensor))

    d_vars = [var for var in tf.trainable_variables() if "d_" in var.name]
    c_vars = [var for var in tf.trainable_variables() if "c_" in var.name]

    opt_classifier = tf.train.AdamOptimizer(0.0001, beta1=0.5).minimize(classification_loss, var_list=c_vars)


    cifar_batch_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    cifar_data, cifar_labels = _read_preprocess_cifar10(train_path, cifar_batch_files)

    cifar_data_train = np.copy(cifar_data)
    cifar_train_labels = np.copy(cifar_labels)
    # cifar_train_labels = np.zeros((cifar_labels.shape[0], 10))
    # cifar_train_labels[np.arange(cifar_labels.shape[0]), cifar_labels] = 1

    img_set = cifar_data_train.shape[0]

    example_idx = np.arange(0, img_set)

    example_idx_rnd = np.random.permutation(example_idx)

    train_idx = example_idx_rnd

    num_iter = int(math.ceil(len(train_idx)/batch_size))

    # saver for discriminator vars only
    saver = tf.train.Saver(var_list = d_vars)

    sess = tf.Session()

    # intialize tf variables
    sess.run(tf.global_variables_initializer())

    chkpt_fname = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, chkpt_fname)

    for epoch_idx in range(0, num_epochs):

        train_idx_shuffle = np.random.permutation(train_idx)

        #train
        for i in range(0, num_iter):
            batch_start = i*batch_size
            batch_end = (i+1)*batch_size

            batch_idx = train_idx_shuffle[batch_start:batch_end]

            batch_labels = cifar_train_labels[batch_idx]

            batch_imgs = cifar_data_train[batch_idx, :, :, :]

            feed_dict_in = {img_data:batch_imgs, label_tensor:batch_labels, is_train:True}

            _, loss = sess.run([opt_classifier, classification_loss], feed_dict = feed_dict_in)

            print loss

    #test
    cifar_batch_files = ["test_batch"]
    cifar_data, cifar_labels = _read_preprocess_cifar10(train_path, cifar_batch_files)
    # cifar_batch_files = ["data_batch_1"]
    # cifar_data, cifar_labels = _read_preprocess_cifar10(train_path, cifar_batch_files)

    test_labels = np.copy(cifar_labels)

    test_imgs = np.copy(cifar_data)

    feed_dict_test = {img_data:test_imgs, label_tensor:test_labels, is_train:False}

    test_loss, test_label_out = sess.run([classification_loss, classifier_labels], feed_dict = feed_dict_test)

    test_acc = np.mean(test_labels == test_label_out, axis = 0)

    print "accuracy: "+str(test_acc)

def main(args):
    #classify_celebA(args)
    classify_cifar(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
