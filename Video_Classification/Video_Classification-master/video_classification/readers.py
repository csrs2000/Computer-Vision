

import tensorflow as tf
import utils

from tensorflow import logging
def resize_axis(tensor, axis, new_size, fill_value=0):
  
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized

class BaseReader(object):
  

  def prepare_reader(self, unused_filename_queue):
   
    raise NotImplementedError()


class YT8MAggregatedFeatureReader(BaseReader):
  

  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["mean_inc3"]):
   

    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names

  def prepare_reader(self, filename_queue, batch_size=1024):
   
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

    tf.add_to_collection("serialized_examples", serialized_examples)
    return self.prepare_serialized_examples(serialized_examples)

  def prepare_serialized_examples(self, serialized_examples):
    # set the mapping from the fields to data types in the proto
    num_features = len(self.feature_names)
    assert num_features > 0, "self.feature_names is empty!"
    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    feature_map = {"id": tf.FixedLenFeature([], tf.string),
                   "labels": tf.VarLenFeature(tf.int64)}
    for feature_index in range(num_features):
      feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
          [self.feature_sizes[feature_index]], tf.float32)

    features = tf.parse_example(serialized_examples, features=feature_map)

    labels = tf.sparse_to_indicator(features["labels"], self.num_classes)
    labels.set_shape([None, self.num_classes])
    concatenated_features = tf.concat([
        features[feature_name] for feature_name in self.feature_names], 1)

    return features["id"], concatenated_features, labels, tf.ones([tf.shape(serialized_examples)[0]])

class YT8MFrameFeatureReader(BaseReader):
  

  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["inc3"],
               max_frames=300):
   

    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.max_frames = max_frames

  def get_video_matrix(self,
                       features,
                       feature_size,
                       max_frames,
                       max_quantized_value,
                       min_quantized_value):
   
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = utils.Dequantize(decoded_features,
                                      max_quantized_value,
                                      min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames

  def prepare_reader(self,
                     filename_queue,
                     max_quantized_value=2,
                     min_quantized_value=-2):
   
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    return self.prepare_serialized_examples(serialized_example,
        max_quantized_value, min_quantized_value)

  def prepare_serialized_examples(self, serialized_example,
      max_quantized_value=2, min_quantized_value=-2):

    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={"id": tf.FixedLenFeature(
            [], tf.string),
                          "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in self.feature_names
        })

    # read ground truth labels
    labels = (tf.cast(
        tf.sparse_to_dense(contexts["labels"].values, (self.num_classes,), 1,
            validate_indices=False),
        tf.bool))

    # loads (potentially) different types of features and concatenates them
    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    num_frames = -1  # the number of frames in the video
    feature_matrices = [None] * num_features  # an array of different features
    for feature_index in range(num_features):
      feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
          features[self.feature_names[feature_index]],
          self.feature_sizes[feature_index],
          self.max_frames,
          max_quantized_value,
          min_quantized_value)
      if num_frames == -1:
        num_frames = num_frames_in_this_feature
      else:
        tf.assert_equal(num_frames, num_frames_in_this_feature)

      feature_matrices[feature_index] = feature_matrix

    # cap the number of frames at self.max_frames
    num_frames = tf.minimum(num_frames, self.max_frames)

    # concatenate different features
    video_matrix = tf.concat(feature_matrices, 1)

    # convert to batch format.
    # TODO: Do proper batch reads to revdmove the IO bottleneck.
    batch_video_ids = tf.expand_dims(contexts["id"], 0)
    batch_video_matrix = tf.expand_dims(video_matrix, 0)
    batch_labels = tf.expand_dims(labels, 0)
    batch_frames = tf.expand_dims(num_frames, 0)

    return batch_video_ids, batch_video_matrix, batch_labels, batch_frames
