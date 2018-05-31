from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import tensorflow as tf

import os

import cv2

from absl import app
from absl import flags
from absl import logging

logging.set_verbosity(tf.logging.DEBUG)
tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', 'data/images/', 'Path to images directory')
flags.DEFINE_string('output_file', 'data/train.tfrecords',
                    'Path to store tfrecords')


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_records(images, out_file):
    writer = tf.python_io.TFRecordWriter(out_file)
    for i in range(len(images)):
        if i % 30 == 0:
            logging.debug('Processes: {}/{}'.format(i, len(images)))

        image = cv2.imread(images[i], cv2.IMREAD_UNCHANGED)
        feature = {'image': bytes_feature(tf.compat.as_bytes(image.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


def main(argv):
    del argv
    config = FLAGS
    logging.info('Collecting images')
    input_files = os.listdir(config.input_dir)
    input_files = map(lambda x: os.path.join(config.input_dir, x), input_files)
    logging.info('Converting images to tfrecords')
    write_records(input_files, config.output_file)


if __name__ == '__main__':
    app.run(main)
