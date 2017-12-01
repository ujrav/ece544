"""
dcgan.py: A DCGAN implementation in Tensorflow.

author: Frank Liu - frank.zijie@gmail.com

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Frank Liu (fzliu) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Frank Liu (fzliu) BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# system imports
import argparse
import os
import random
# import shutil

# library imports
import numpy as np
from skimage.color import gray2rgb
# from skimage.io import imread, imsave
from skimage.transform import resize
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import scipy.misc
import cPickle

# local imports
from sdcgan.models import generator, discriminator


# size of the input latent space
Z_SIZE = 100

# training parameters
TRAIN_RATIO = 10
DISPLAY_LOSSES = 50

#Manju: Directory tree:
#dcgan.py
#models.py
#data/celebA/*.jpg
#output/epoch*.jpg   #sampled images after every epoch will be created here.
#output/checkpoint/ #have this directory tree. # Checkpoint files will be created here.

#Please clear out the checkpoint files yourself manually if you want to start fresh again.
#this code cannot cleanup files. 


# directory to snapshot models and images
OUTPUT_PATH = "gs://juman/sdcgan/output1"  # Manju: change this to your bucket. Make sure you have the appropriate folders. This code cannot make directories.
CHECKPOINT_NAME = "checkpoint/dcgan.tfmodel" # make sure you have the checkpoint directory inside output folder.


# argparse
parser = argparse.ArgumentParser(description="Train a DCGAN using Tensorflow.")
parser.add_argument("-n", "--num-epochs", type=int, default=100, help="number of epochs")
parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size to use")
parser.add_argument("-l", "--learning-rate", type=float, default=6e-4, help="generator learning rate")
parser.add_argument("-i", "--image-size", type=int, default=32, help="(square) image size")
parser.add_argument("-s", "--scale-size", type=int, default=32, help="resize length for center crop")
parser.add_argument("-t", "--train-dir", type=str, help="directory to pull training images from")
parser.add_argument("-o", "--output-dir", type=str, default="outputs", help="directory to output generations")
parser.add_argument("-r", "--restore", action="store_true", help="specify to use the latest checkpoint")
parser.add_argument("-z", "--job-dir", type=str, default="outputs", help="blah")

#Manju: I've commented the following because making/ deleting directoris is a pain to do in GCS. So please do it manually.
# def _clean_directory(path):
#     """
#         Clears (and creates) a directory on the filesystem.
#     """

#     if os.path.exists(path):
#         shutil.rmtree(os.path.join(path))
#     os.mkdir(path)


def _sigmoid_loss(logits, targets):
    """
        Wrapper around Tensorflow's sigmoid loss function.
    """

    loss_comp = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets,logits=logits)

    return tf.reduce_mean(loss_comp)
    

def _read_and_preprocess(paths, scale_len, crop_len):
    """
        Reads multiple images (and labels).
    """

    # imgs = []

    # for path in paths:
        
    #     file=file_io.FileIO(path,mode='r')
    #     img = scipy.misc.imread(file,mode='RGB')

    #     # force 3-channel images
    #     # if img.ndim == 2:
    #     #     img = gray2rgb(img)
    #     # elif img.shape[2] == 4:
    #     #     img = img[:, :, :3]

    #     # compute the resize dimension
    #     resize_f = float(scale_len) / min(np.array(img).shape[:2])
    #     new_dims = (int(np.round(img.shape[0] * resize_f)),
    #                 int(np.round(img.shape[1] * resize_f)))

    #     # prevent the input image from blowing up
    #     # factor of 2 is more or less an arbitrary number
    #     max_dim = 2 * scale_len
    #     new_dims = (min(new_dims[0], max_dim),
    #                 min(new_dims[1], max_dim))

    #     # resize and center crop
    #     img = resize(img, new_dims)
    #     top = int(np.ceil((img.shape[0] - crop_len) / 2.0))
    #     left = int(np.ceil((img.shape[1] - crop_len) / 2.0))
    #     img = img[top:(top+crop_len), left:(left+crop_len)]

    #     # preprocessing (tanh)
    #     img = img*2 - 1

    #     imgs.append(img)


    imgs = []
    img=np.random.uniform(-1,1,size=(64,64,3))

    for path in paths:
        
        try:

            file=file_io.FileIO(path,mode='r')
            img = scipy.misc.imread(file,mode='RGB')

            # force 3-channel images
            if np.array(img).ndim == 2:
                img = gray2rgb(img)
            elif np.array(img).ndim!=3:
                img=np.random.uniform(-1,1,size=(64,64,3)) 
                print("Adding noise")
                # when there is a file IO error

            # compute the resize dimension
            

            resize_f = float(scale_len) / min(np.array(img).shape[:2])
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
        
        except:
            imgs.append(img)



    return np.array(imgs)


def _deprocess_and_save(batch_res, epoch, grid_shape=(8, 8), grid_pad=5):
    """
        Deprocesses the generator output and saves the results.
    """

    # create an output grid to hold the images
    (img_h, img_w) = batch_res.shape[1:3]
    grid_h = img_h * grid_shape[0] + grid_pad * (grid_shape[0] - 1)
    grid_w = img_w * grid_shape[1] + grid_pad * (grid_shape[1] - 1)
    img_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # loop through all generator outputs
    for i, res in enumerate(batch_res):
        if i >= grid_shape[0] * grid_shape[1]:
            break

        # deprocessing (tanh)
        img = (res + 1) * 127.5
        img = img.astype(np.uint8)

        # add the image to the image grid
        row = (i // grid_shape[0]) * (img_h + grid_pad)
        col = (i % grid_shape[1]) * (img_w + grid_pad)

        img_grid[row:row+img_h, col:col+img_w, :] = img

    # save the output image
    fname = "epoch{0}.jpg".format(epoch) if epoch >= 0 else "result.jpg"
    fileo=file_io.FileIO(os.path.join(OUTPUT_PATH, fname),mode='w+')
    scipy.misc.imsave(fileo, img_grid)

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

    cifar_data = cifar_data*2 - 1

    return cifar_data, cifar_labels



def train_dcgan(n_epochs, batch_size, lr_rate, crop_len, scale_len, restore, train_path):
    """
        Train DCGAN.

        :param int n_epochs:
            Total number of epochs over the input data to train for.

        :param int batch_size:
            Batch size to use for training.

        :param float lr_rate:
            Generator learning rate.

        :param int crop_len:
            Image side length to use.

        :param int scale_len:
            Amount to scale the minimum side length to (for augmentation).

        :param bool restore:
            Specifies whether or not the latest checkpoint should be used.

        :param string data_path:
            path to cifar data.
    """

    assert scale_len >= crop_len, "invalid resize or crop length"

    # create placeholders
    sample = tf.placeholder(tf.float32, shape=[batch_size, Z_SIZE], name="sample")
    real = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3], name="real")
    is_train = tf.placeholder(tf.bool, name="is_train")

    # instantiate the models
    G = generator(sample, is_train, crop_len)
    D_fake = discriminator(G, is_train,reuse=False)
    
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        tf.get_variable_scope().reuse_variables()
        D_real = discriminator(real, is_train,reuse=True)


    # create losses
    loss_G = _sigmoid_loss(D_fake, tf.ones_like(D_fake))
    loss_D = _sigmoid_loss(D_fake, tf.zeros_like(D_fake)) + \
            _sigmoid_loss(D_real, tf.ones_like(D_real))

    # acquire tensors for generator and discriminator
    # trick from carpedm20's implementation on github
    g_vars = [var for var in tf.trainable_variables() if "g_" in var.name]
    d_vars = [var for var in tf.trainable_variables() if "d_" in var.name]

    # create optimization objectives
    global_step = tf.Variable(0, name="global_step", trainable=False)
    opt_G = tf.train.AdamOptimizer(lr_rate, beta1=0.7).minimize(loss_G, var_list=g_vars)
    opt_D = tf.train.AdamOptimizer(6e-4, beta1=0.7).minimize(loss_D, var_list=d_vars)

    # create a saver and restore variables, if necessary
    saver = tf.train.Saver()
    model_path = os.path.join(OUTPUT_PATH, CHECKPOINT_NAME)

    # do some initialization
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    if restore:
        chkpt_fname = tf.train.latest_checkpoint(os.path.join(OUTPUT_PATH, 'checkpoint'))
        saver.restore(sess, chkpt_fname)
        # _clean_directory(OUTPUT_PATH)

    # reference vector (to examine epoch-to-epoch changes)
    vec_ref = np.random.uniform(-1, 1,size=(batch_size, Z_SIZE)).astype(np.float32)

    cifar_batch_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    cifar_data, cifar_labels = _read_preprocess_cifar10(train_path, cifar_batch_files)

    cifar_data_train = cifar_data

    # begin training
    n_iterations = cifar_data.shape[0] // batch_size
    for epoch in range(sess.run(global_step), n_epochs):
        print("------- EPOCH {0} -------".format(epoch))
        epoch_data = np.random.permutation(cifar_data)

        # train the discriminator for one epoch
        for i in range(n_iterations):

            offset = (i * batch_size) % cifar_data.shape[0]
            imgs = epoch_data[offset:(offset + batch_size),:,:,:]
            vec = np.random.uniform(-1, 1,size=(batch_size, Z_SIZE)).astype(np.float32)

            # minimize discriminator loss
            sess.run(opt_D, feed_dict={real: imgs, sample: vec, is_train: True})

            # minimize generator loss once or twice
            
            sess.run(opt_G, feed_dict={sample: vec, is_train: True})
            # sess.run(opt_G, feed_dict={sample: vec, is_train: True}) #Manju: Some papers suggest to run generator twice. I've seen mixed results with this. Not quite sure what to do
            

            # log the error
            if i % DISPLAY_LOSSES == 0:
                err_G = sess.run(loss_G, feed_dict={sample: vec, is_train: False})
                err_D = sess.run(loss_D, feed_dict={real: imgs, sample: vec, is_train: False})
                print("  Iteration {0}".format(i))
                print("    generator loss = {0}".format(err_G))
                print("    discriminator loss = {0}".format(err_D))

        # save the model and sample results at the end of each epoch
        sess.run(tf.assign(global_step, epoch + 1))
        saver.save(sess, model_path, global_step=global_step)
        batch_res = sess.run(G, {sample: vec_ref, is_train: False})
        _deprocess_and_save(batch_res, epoch)


# Manju: Commented the following because it doesnt work, and it is not needed. Dont sample it separately, just run the code again with --restore to get new 
# samples after an epoch

# def sample_dcgan(side_len,batch_size):
#     """
#         Runs inference on the generator.

#         :param int side_len:
#             Side length for generator output (must match `crop_len`).
#     """

#     # only need to restore the generator's variables
#     sample = tf.placeholder(tf.float32, shape=[batch_size, Z_SIZE], name="sample")

#     with tf.variable_scope(tf.get_variable_scope()) as scope:
#         tf.get_variable_scope().reuse_variables()
#         G = generator(sample, None, side_len)

#     sess = tf.Session()
#     sess.run(tf.initialize_all_variables())

#     # apply the trained generator weights
#     saver = tf.train.Saver()
#     model_path = os.path.join(OUTPUT_PATH, 'checkpoint')
#     chkpt_fname = tf.train.latest_checkpoint(model_path)
#     saver.restore(sess, chkpt_fname)

#     # run a forward pass
#     vec = np.random.normal(size=(batch_size, Z_SIZE))
#     result = sess.run(G, {sample: vec})

#     # save the result
#     _deprocess_and_save(result, -1)


def main(args):
    """
        Entry point.
    """

    global OUTPUT_PATH

    # should probably be passing this to `train_dcgan()` instead
    if args.output_dir:
        OUTPUT_PATH = args.output_dir



    train_dcgan(args.num_epochs, args.batch_size, args.learning_rate, 
                args.image_size, args.scale_size, args.restore, args.train_dir)  

        
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
