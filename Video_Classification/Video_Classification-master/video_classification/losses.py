

import tensorflow as tf
from tensorflow import flags
import scipy.io as sio
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_float(
      "alpha", "0.5",
      "Ponderation for XENT")


class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    
    raise NotImplementedError()


class CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      alpha = FLAGS.alpha

      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = 2*(alpha*float_labels * tf.log(predictions + epsilon) + (1-alpha)*(
          1 - float_labels) * tf.log(1 - predictions + epsilon))
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))
