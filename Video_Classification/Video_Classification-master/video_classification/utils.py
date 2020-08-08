


import numpy
import tensorflow as tf
from tensorflow import logging


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
 
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias


def MakeSummary(name, value):
  
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary


def AddGlobalStepSummary(summary_writer,
                         global_step_val,
                         global_step_info_dict,
                         summary_scope="Eval"):
  
  this_hit_at_one = global_step_info_dict["hit_at_one"]
  this_perr = global_step_info_dict["perr"]
  this_loss = global_step_info_dict["loss"]
  examples_per_second = global_step_info_dict.get("examples_per_second", -1)

  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@1", this_hit_at_one),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Perr", this_perr),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Loss", this_loss),
      global_step_val)

  if examples_per_second != -1:
    summary_writer.add_summary(
        MakeSummary("GlobalStep/" + summary_scope + "_Example_Second",
                    examples_per_second), global_step_val)

  summary_writer.flush()
  info = ("global_step {0} | Batch Hit@1: {1:.3f} | Batch PERR: {2:.3f} | Batch Loss: {3:.3f} "
          "| Examples_per_sec: {4:.3f}").format(
              global_step_val, this_hit_at_one, this_perr, this_loss,
              examples_per_second)
  return info


def AddEpochSummary(summary_writer,
                    global_step_val,
                    epoch_info_dict,
                    summary_scope="Eval"):
 
  epoch_id = epoch_info_dict["epoch_id"]
  avg_hit_at_one = epoch_info_dict["avg_hit_at_one"]
  avg_perr = epoch_info_dict["avg_perr"]
  avg_loss = epoch_info_dict["avg_loss"]
  aps = epoch_info_dict["aps"]
  gap = epoch_info_dict["gap"]
  mean_ap = numpy.mean(aps)

  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@1", avg_hit_at_one),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Perr", avg_perr),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Loss", avg_loss),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_MAP", mean_ap),
          global_step_val)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_GAP", gap),
          global_step_val)
  summary_writer.flush()

  info = ("epoch/eval number {0} | Avg_Hit@1: {1:.3f} | Avg_PERR: {2:.3f} "
          "| MAP: {3:.3f} | GAP: {4:.4f} | Avg_Loss: {5:3f}").format(
          epoch_id, avg_hit_at_one, avg_perr, mean_ap, gap, avg_loss)
  return info

def GetListOfFeatureNamesAndSizes(feature_names, feature_sizes):
 
  list_of_feature_names = [
      feature_names.strip() for feature_names in feature_names.split(',')]
  list_of_feature_sizes = [
      int(feature_sizes) for feature_sizes in feature_sizes.split(',')]
  if len(list_of_feature_names) != len(list_of_feature_sizes):
    logging.error("length of the feature names (=" +
                  str(len(list_of_feature_names)) + ") != length of feature "
                  "sizes (=" + str(len(list_of_feature_sizes)) + ")")

  return list_of_feature_names, list_of_feature_sizes

