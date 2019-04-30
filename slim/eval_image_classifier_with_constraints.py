# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as c
import cPickle as pkl
import functools as ft
import json
import math
import os
import os.path as osp
import sys
import tensorflow as tf

from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.training import saver as tf_saver

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory


slim = tf.contrib.slim
cwd = osp.abspath(os.getcwd())


sys.path.append("/olli/tensorflow/code")
import tensor_tools as tt
import numpy as np


def make_conv2d_noisy(sigma):
    """
        Make conv2d noisy by monkey patching the method (this way we don't miss
        any invocations).
    """
    print("Noising every relu with sigma of {:g}.".format(sigma))

    def noisy_relu(tensor):
        return tf.nn.relu(tensor +\
            tf.truncated_normal(tf.shape(tensor), mean=0., stddev=sigma))

    tf_conv2d = slim.conv2d

    @ft.wraps(tf_conv2d)
    def conv2d(*args, **kwargs):
        print("Conv2D called with:")
        print(args)
        print(kwargs)

        if "activation_fn" in kwargs:
            orig_fn = kwargs["activation_fn"]
            #  raise ValueError("Found custom activation function!")
            act_fn = lambda x: noisy_relu(orig_fn(x))
        else:
            act_fn = noisy_relu

        kwargs["activation_fn"] = act_fn

        return tf_conv2d(*args, **kwargs)

    setattr(slim, "conv2d", conv2d)


def make_conv2d_noisy_fixed(sigma):
    """
        Make conv2d noisy by monkey patching the method (this way we don't miss
        any invocations).
    """
    print("Noising every relu (fixed) with sigma of {:g}.".format(sigma))

    def noisy_relu_fixed(tensor):
        shape = [1,] + tensor.shape.as_list()[1:]
        return tf.nn.relu(tensor + tt.truncated_gauss(shape, sigma=sigma))
        #  return tf.nn.relu(tensor +
            #  tf.truncated_normal(tf.shape(tensor), mean=0., stddev=sigma,
                #  seed=np.random.randint(np.iinfo(np.int32).max)))

    tf_conv2d = slim.conv2d

    @ft.wraps(tf_conv2d)
    def conv2d(*args, **kwargs):
        print("Conv2D called with:")
        print(args)
        print(kwargs)

        if "activation_fn" in kwargs:
            orig_fn = kwargs["activation_fn"]
            #  raise ValueError("Found custom activation function!")
            act_fn = lambda x: noisy_relu_fixed(orig_fn(x))
        else:
            act_fn = noisy_relu_fixed

        kwargs["activation_fn"] = act_fn

        return tf_conv2d(*args, **kwargs)

    setattr(slim, "conv2d", conv2d)


def make_conv2d_quantized_indv(num_bits, name_to_max):
    """
        Quantize tensors after ReLU, each layer with an individual max value.
    """
    print("Quantizing every relu with {} bits ".format(num_bits))

    num_values = 2**num_bits
    max_index = num_values - 1

    def quantize_indv_relu(tensor):
        relu = tf.nn.relu(tensor)
        name = relu.name[:-2]

        if name not in name_to_max:
            print("WARNING: no recorded max value for {}..".format(name))
            return relu
        step = name_to_max[name] / max_index

        quant = tf.div(relu, step)
        round = tf.rint(quant)
        clip = tf.minimum(round, max_index)

        return tf.multiply(clip, step)

    tf_conv2d = slim.conv2d

    @ft.wraps(tf_conv2d)
    def conv2d(*args, **kwargs):
        print("Conv2D called with:")
        print(args)
        print(kwargs)

        if "activation_fn" in kwargs:
            orig_fn = kwargs["activation_fn"]
            #  raise ValueError("Found custom activation function!")
            act_fn = lambda x: quantize_indv_relu(orig_fn(x))
        else:
            act_fn = quantize_indv_relu

        kwargs["activation_fn"] = act_fn

        return tf_conv2d(*args, **kwargs)

    setattr(slim, "conv2d", conv2d)


def make_conv2d_quantized_global(num_bits, max_value):
    """
        Quantize tensors after ReLU, each layer with an individual max value.
    """
    assert max_value > 0.
    print("Quantizing every relu with {} bits and one max value of {}".format(
        num_bits, max_value))

    num_values = 2**num_bits
    max_index = num_values - 1

    step = max_value / max_index

    def quantize_global_relu(tensor):
        relu = tf.nn.relu(tensor)
        quant = tf.div(relu, step)
        round = tf.rint(quant)
        clip = tf.minimum(round, max_index)
        return tf.multiply(clip, step)

    tf_conv2d = slim.conv2d

    @ft.wraps(tf_conv2d)
    def conv2d(*args, **kwargs):
        print("Conv2D called with:")
        print(args)
        print(kwargs)

        if "activation_fn" in kwargs:
            orig_fn = kwargs["activation_fn"]
            #  raise ValueError("Found custom activation function!")
            act_fn = lambda x: quantize_global_relu(orig_fn(x))
        else:
            act_fn = quantize_global_relu

        kwargs["activation_fn"] = act_fn

        return tf_conv2d(*args, **kwargs)

    setattr(slim, "conv2d", conv2d)


def make_conv2d_limit_output(global_max):
    """
        Quantize tensors after ReLU, each layer with an individual max value.
    """
    print("Limiting output of every ReLU to {:g}.".format(global_max))

    def limit_relu(tensor):
        return tf.minimum(tf.nn.relu(tensor), global_max)

    tf_conv2d = slim.conv2d

    @ft.wraps(tf_conv2d)
    def conv2d(*args, **kwargs):
        print("Conv2D called with:")
        print(args)
        print(kwargs)

        if "activation_fn" in kwargs:
            orig_fn = kwargs["activation_fn"]
            #  raise ValueError("Found custom activation function!")
            act_fn = lambda x: limit_relu(orig_fn(x))
        else:
            act_fn = limit_relu

        kwargs["activation_fn"] = act_fn

        return tf_conv2d(*args, **kwargs)

    setattr(slim, "conv2d", conv2d)


def make_conv2d_track_range():
    """
        Make conv2d noisy by monkey patching the method (this way we don't miss
        any invocations).
    """
    print("Tracking every ReLU range.")

    tf_conv2d = slim.conv2d

    tracked_relus = []
    name_to_max = {}

    @ft.wraps(tf_conv2d)
    def conv2d(*args, **kwargs):
        print("Conv2D called with:")
        print(args)
        print(kwargs)

        def tracking_relu(tensor):
            relu = tf.nn.relu(tensor)
            tracked_relus.append(relu)
            name_to_max[relu.name[:-2]] = tf.reduce_max(relu)
            return relu

        if "activation_fn" in kwargs:
            orig_fn = kwargs["activation_fn"]
            #  raise ValueError("Found custom activation function!")
            act_fn = lambda x: tracking_relu(orig_fn(x))
        else:
            act_fn = tracking_relu

        kwargs["activation_fn"] = act_fn

        return tf_conv2d(*args, **kwargs)

    setattr(slim, "conv2d", conv2d)

    return tracked_relus, name_to_max


def get_max(tracked_relus):
    return tf.reduce_max(tf.stack(map(tf.reduce_max, tracked_relus)))


tf.app.flags.DEFINE_bool(
    'get_range', False,
    'Get the range of all relus.')

tf.app.flags.DEFINE_float(
    'limit_relu', 0.0, 'If > 0.0 the ouput of every ReLU is limited to this.')

tf.app.flags.DEFINE_float(
    'max_value_global', 0.0, 'Global max value for quantization.')

tf.app.flags.DEFINE_string(
    'max_values_path',
    '/olli/tensorflow/data/inception_v4_imagenet_relu_ranges.json',
    'JSON file in which the max values for each relu layer are stored.')

tf.app.flags.DEFINE_integer(
    'num_bits', 32,
    'Number of bits for quantization.')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'Whether or not to quantize ReLUs.')

tf.app.flags.DEFINE_bool(
    'quantize_individually', False,
    'Whether or not to quantize each ReLU layer by its own max value.')

tf.app.flags.DEFINE_string(
    'ranges_path', None, 'Path under which to dump the extracted ranges.')

tf.app.flags.DEFINE_bool(
    'relu_noise_fixed', False,
    'Noise applied to each relu fixed at creation.')

tf.app.flags.DEFINE_float(
    'sigma', 0.0,
    'Noise applied to each relu.')

tf.app.flags.DEFINE_integer(
    'seed_numpy', 42424242,
    'Seed for numpy.')

#################
# DEFAULT FLAGS #
#################

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  np.random.seed(FLAGS.seed_numpy)

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ######################
    # apply custom logic #
    ######################
    if FLAGS.quantize:
        if FLAGS.quantize_individually:
            with open(FLAGS.max_values_path, "r") as f:
                names_to_max = json.load(f)
            make_conv2d_quantized_indv(FLAGS.num_bits, names_to_max)
        else:
            make_conv2d_quantized_global(
                FLAGS.num_bits, FLAGS.max_value_global)

    if FLAGS.limit_relu > 0:
        make_conv2d_limit_output(FLAGS.limit_relu)

    if FLAGS.sigma > 0:
        if FLAGS.relu_noise_fixed:
            make_conv2d_noisy_fixed(FLAGS.sigma)
        else:
            make_conv2d_noisy(FLAGS.sigma)

    tracked_relus = []
    if FLAGS.get_range:
        tracked_relus, name_to_max = make_conv2d_track_range()

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    metrics_map = {
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    } 
    if FLAGS.get_range:
        metrics_map["maximum"] = slim.metrics.streaming_concat(
            tf.reshape(get_max(tracked_relus), (1,)))

        for k, v in name_to_max.iteritems():
            metrics_map[k] = slim.metrics.streaming_concat(tf.reshape(v, (1,)))

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
            metrics_map)

    # Print the summaries to screen.
    for name, value in names_to_values.iteritems():
      summary_name = 'eval/%s' % name
      #  if name == "maximum":
      if name not in ["Accuracy", "Recall_5"]:
          value = tf.reduce_max(value)
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    #  if FLAGS.get_range:
      #  name = "maximum"
      #  value = max_value = get_max(tracked_relus)

      #  summary_name = 'eval/%s' % name
      #  op = tf.summary.scalar(summary_name, value,
              #  collections=[])
      #  op = tf.Print(op, [value], summary_name)
      #  tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
        num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    eval_op = list(names_to_updates.values()) 

    #  if not FLAGS.get_range:
    #  from pudb import set_trace; set_trace()
    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=eval_op,
        variables_to_restore=variables_to_restore)

    #  else:
        #  summary_op = summary.merge_all()

        #  saver = tf_saver.Saver(variables_to_restore)

        #  writer = tf.summary.FileWriter(osp.dirname(FLAGS.ranges_path))

if __name__ == '__main__':
  tf.app.run()
