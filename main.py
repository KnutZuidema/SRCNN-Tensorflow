from model import SRCNN
from utils import input_setup

import numpy as np
import tensorflow as tf
import argparse

import pprint
import os

args = argparse.ArgumentParser()
args.add_argument("--epoch", default=15000, help="Number of epoch [15000]")
args.add_argument("--batch_size", default=128, help="The size of batch images [128]")
args.add_argument("--image_size", default=33, help="The size of image to use [33]")
args.add_argument("--label_size", default=21, help="The size of label to produce [21]")
args.add_argument("--learning_rate", default=1e-4, help="The learning rate of gradient descent algorithm [1e-4]")
args.add_argument("--c_dim", default=1, help="Dimension of image color. [1]")
args.add_argument("--scale", default=3, help="The size of scale factor for preprocessing input image [3]")
args.add_argument("--stride", default=14, help="The size of stride to apply input image [14]")
args.add_argument("--checkpoint_dir", default="checkpoint", help="Name of checkpoint directory [checkpoint]")
args.add_argument("--sample_dir", default="sample", help="Name of sample directory [sample]")
args.add_argument("--is_train", default=True, help="True for training, False for testing [True]")
ARGS = args.parse_args()

pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(ARGS)

    if not os.path.exists(ARGS.checkpoint_dir):
        os.makedirs(ARGS.checkpoint_dir)
    if not os.path.exists(ARGS.sample_dir):
        os.makedirs(ARGS.sample_dir)

    with tf.Session() as sess:
        srcnn = SRCNN(sess,
                      image_size=ARGS.image_size,
                      label_size=ARGS.label_size,
                      batch_size=ARGS.batch_size,
                      c_dim=ARGS.c_dim,
                      checkpoint_dir=ARGS.checkpoint_dir,
                      sample_dir=ARGS.sample_dir)

        srcnn.train(ARGS)


if __name__ == '__main__':
    tf.app.run()
