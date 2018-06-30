from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from absl import app
from absl import flags
from absl import logging

from model import Discriminator, Generator
from buffer import RingBuffer

logging.set_verbosity(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', 'data/train.tfrecords', 'Path to real data')

flags.DEFINE_float('generator_learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_integer('iterations', 150000, 'Number of trainig iterations')
flags.DEFINE_integer('seed', 971,
                     'Seed to feed the tensorflow graph and numpy state')
flags.DEFINE_integer('G_steps', 1,
                     'Number of concurrent steps to run Generator')
flags.DEFINE_integer('D_steps', 5,
                     'Number of concurrent steps to run Discriminatro')
flags.DEFINE_integer('buffer_size', 30, 'Size of the accuracy buffer')

flags.DEFINE_string('checkpoint_directory', 'checkpoints/',
                    'Directory to read and write checkpoints')
flags.DEFINE_string('summary_directory', 'summaries/',
                    'Directory to write summaries')
flags.DEFINE_bool('save_checkpoints', True, 'Whether to save checkpoints')
flags.DEFINE_bool('save_summaries', True, 'Whether to save summaries')
flags.DEFINE_bool('restore', True, 'Whether to restore checkpoints (if exists)')
flags.DEFINE_integer('checkpoint_save_secs', None,
                     'How often to save checkpoints')
flags.DEFINE_integer('summary_save_secs', None, 'How often to save summaries')
flags.DEFINE_integer('checkpoint_save_steps', 500,
                     'How often to save checkpoints')
flags.DEFINE_integer('summary_save_steps', 100, 'How often to save summaries')
flags.DEFINE_integer('log_step', 100, 'How often to write console logs')
flags.DEFINE_integer('image_size', 128, 'Rescaled image')


def _parse_record(example, image_size):
    features = {
        'image': tf.FixedLenFeature([], tf.string),
    }
    parsed_feature = tf.parse_single_example(example, features)
    image = parsed_feature['image']
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [215, 215, 4])
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, [image_size, image_size])
    image = image / 255.0

    return image


def accuracy(predictions, labels):
    return tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))


def main(argv):
    del argv
    config = FLAGS

    tf.set_random_seed(config.seed)
    np_state = np.random.RandomState(config.seed)

    global_step = tf.train.get_or_create_global_step()
    global_step_update = tf.assign(global_step, global_step + 1)

    real_ds = tf.data.TFRecordDataset(config.input_path)
    real_ds = real_ds.map(lambda x: _parse_record(x, config.image_size))
    real_ds = real_ds.shuffle(buffer_size=1000)
    real_ds = real_ds.batch(config.batch_size // 2)    # Half will be generated
    real_ds = real_ds.repeat()
    real_ds_iterator = real_ds.make_one_shot_iterator()
    real_ds_example = real_ds_iterator.get_next()

    discriminator = Discriminator('discriminator')
    generator = Generator('generator')

    z = tf.placeholder(dtype=tf.float32, shape=[None, 100])

    G_sample = generator.create_main_graph(z)

    D_logit_real = discriminator.create_main_graph(real_ds_example)
    D_logit_fake = discriminator.create_main_graph(G_sample)

    D_expected_real = tf.zeros_like(D_logit_real)
    D_expected_fake = tf.ones_like(D_logit_fake)

    D_loss_real = tf.losses.sigmoid_cross_entropy(
        D_expected_real, D_logit_real, label_smoothing=0.2)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(
        D_expected_fake, D_logit_fake, label_smoothing=0.00)

    D_loss = 0.5 * (D_loss_real + D_loss_fake)

    G_loss = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(D_logit_fake), D_logit_fake, label_smoothing=0.00)

    with tf.variable_scope('metrics'):
        D_prediction_real = tf.round(tf.nn.sigmoid(D_logit_real))
        D_prediction_fake = tf.round(tf.nn.sigmoid(D_logit_fake))

        D_accuracy_real = accuracy(D_prediction_real, D_expected_real)
        D_accuracy_fake = accuracy(D_prediction_fake, D_expected_fake)

        real_size = tf.to_float(tf.shape(D_prediction_real)[0])
        fake_size = tf.to_float(tf.shape(D_prediction_fake)[0])
        D_accuracy = (real_size * D_accuracy_real + fake_size * D_accuracy_fake
                     ) / (real_size + fake_size)

    update_ops = tf.get_collection(
        tf.GraphKeys.UPDATE_OPS, scope='discriminator')
    with tf.control_dependencies(update_ops):
        D_optimizer = tf.train.AdamOptimizer(
            config.discriminator_learning_rate).minimize(
                D_loss, var_list=discriminator.get_variables())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    with tf.control_dependencies(update_ops):
        G_optimizer = tf.train.AdamOptimizer(
            config.generator_learning_rate).minimize(
                G_loss, var_list=generator.get_variables())

    with tf.variable_scope('summaries'):
        D_loss_summary = tf.summary.scalar(
            'loss', D_loss, family='discriminator')
        D_accuracy_real_summary = tf.summary.scalar(
            'real_accuracy', D_accuracy_real, family='discriminator')
        D_accuracy_fake_summary = tf.summary.scalar(
            'fake_accuracy', D_accuracy_fake, family='discriminator')
        D_accuracy_summary = tf.summary.scalar(
            'accuracy', D_accuracy, family='discriminator')
        G_loss_summary = tf.summary.scalar('loss', G_loss, family='generator')
        G_image_summary = tf.summary.image(
            'generation', G_sample, max_outputs=1, family='generator')
        Real_image_summary = tf.summary.image(
            'real', real_ds_example, max_outputs=1)

        summary_op = tf.summary.merge_all()

    # Session
    hooks = []
    hooks.append(tf.train.StopAtStepHook(num_steps=config.iterations))
    if (config.save_checkpoints):
        hooks.append(
            tf.train.CheckpointSaverHook(
                checkpoint_dir=config.checkpoint_directory,
                save_secs=config.checkpoint_save_secs,
                save_steps=config.checkpoint_save_steps))

    if (config.save_summaries):
        hooks.append(
            tf.train.SummarySaverHook(
                output_dir=config.summary_directory,
                save_secs=config.summary_save_secs,
                save_steps=config.summary_save_steps,
                summary_op=summary_op))

    if config.restore:
        sess = tf.train.MonitoredTrainingSession(
            checkpoint_dir=config.checkpoint_directory,
            save_checkpoint_steps=None,
            save_checkpoint_secs=None,
            save_summaries_steps=None,
            save_summaries_secs=None,
            log_step_count_steps=None,
            hooks=hooks)
    else:
        sess = tf.train.MonitoredTrainingSession(
            save_checkpoint_steps=None,
            save_checkpoint_secs=None,
            save_summaries_steps=None,
            save_summaries_secs=None,
            log_step_count_steps=None,
            hooks=hooks)

    def step_generator(step_context, accuracy_buffer):
        np_global_step = step_context.session.run(global_step)
        step_context.session.run(global_step_update)

        random_noise = np_state.normal(size=[config.batch_size, 100])
        _, np_loss, np_accuracy = step_context.run_with_hooks(
            [G_optimizer, G_loss, D_accuracy], feed_dict={
                z: random_noise
            })

        accuracy_buffer.add(np_accuracy)
        if np_global_step % config.log_step == 0:
            logging.debug(
                'Training Generator: Step: {}   Loss: {:.3e}   Accuracy: {:.2f}'.
                format(np_global_step, np_loss,
                       accuracy_buffer.mean() * 100))

    def step_discriminator(step_context, accuracy_buffer):
        np_global_step = step_context.session.run(global_step)
        step_context.session.run(global_step_update)

        random_noise = np_state.normal(size=[config.batch_size // 2, 100])
        _, np_loss, np_accuracy = step_context.run_with_hooks(
            [D_optimizer, D_loss, D_accuracy], feed_dict={
                z: random_noise
            })

        accuracy_buffer.add(np_accuracy)
        if np_global_step % config.log_step == 0:
            logging.debug(
                'Training Discriminator: Step: {}   Loss Mean: {:.3e}   Accuracy: {:.2f}'.
                format(np_global_step, np_loss,
                       accuracy_buffer.mean() * 100))

    accuracy_buffer = RingBuffer(config.buffer_size)
    accuracy_buffer.clear()
    while not sess.should_stop():
        for _ in xrange(config.D_steps):
            sess.run_step_fn(
                lambda step_context: step_discriminator(step_context, accuracy_buffer)
            )
        for _ in xrange(config.G_steps):
            sess.run_step_fn(
                lambda step_context: step_generator(step_context, accuracy_buffer)
            )

    sess.close()


if __name__ == '__main__':
    app.run(main)
