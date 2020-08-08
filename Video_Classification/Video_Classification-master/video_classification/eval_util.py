import datetime
import numpy
import mean_average_precision_calculator as map_calculator
import average_precision_calculator as ap_calculator

def flatten(l):
  """ Merges a list of lists into a single list. """
  return [item for sublist in l for item in sublist]

def calculate_hit_at_one(predictions, actuals):
 
  top_prediction = numpy.argmax(predictions, 1)
  hits = actuals[numpy.arange(actuals.shape[0]), top_prediction]
  return numpy.average(hits)


def calculate_precision_at_equal_recall_rate(predictions, actuals):
  
  aggregated_precision = 0.0
  num_videos = actuals.shape[0]
  for row in numpy.arange(num_videos):
    num_labels = int(numpy.sum(actuals[row]))
    top_indices = numpy.argpartition(predictions[row],
                                     -num_labels)[-num_labels:]
    item_precision = 0.0
    for label_index in top_indices:
      if predictions[row][label_index] > 0:
        item_precision += actuals[row][label_index]
    item_precision /= top_indices.size
    aggregated_precision += item_precision
  aggregated_precision /= num_videos
  return aggregated_precision

def calculate_gap(predictions, actuals, top_k=20):
 
  gap_calculator = ap_calculator.AveragePrecisionCalculator()
  sparse_predictions, sparse_labels, num_positives = top_k_by_class(predictions, actuals, top_k)
  gap_calculator.accumulate(flatten(sparse_predictions), flatten(sparse_labels), sum(num_positives))
  return gap_calculator.peek_ap_at_n()


def top_k_by_class(predictions, labels, k=20):
  
  if k <= 0:
    raise ValueError("k must be a positive integer.")
  k = min(k, predictions.shape[1])
  num_classes = predictions.shape[1]
  prediction_triplets= []
  for video_index in range(predictions.shape[0]):
    prediction_triplets.extend(top_k_triplets(predictions[video_index],labels[video_index], k))
  out_predictions = [[] for v in range(num_classes)]
  out_labels = [[] for v in range(num_classes)]
  for triplet in prediction_triplets:
    out_predictions[triplet[0]].append(triplet[1])
    out_labels[triplet[0]].append(triplet[2])
  out_true_positives = [numpy.sum(labels[:,i]) for i in range(num_classes)]

  return out_predictions, out_labels, out_true_positives

def top_k_triplets(predictions, labels, k=20):
  
  m = len(predictions)
  k = min(k, m)
  indices = numpy.argpartition(predictions, -k)[-k:]
  return [(index, predictions[index], labels[index]) for index in indices]

class EvaluationMetrics(object):
  """A class to store the evaluation metrics."""

  def __init__(self, num_class, top_k):
   
    self.sum_hit_at_one = 0.0
    self.sum_perr = 0.0
    self.sum_loss = 0.0
    self.map_calculator = map_calculator.MeanAveragePrecisionCalculator(num_class)
    self.global_ap_calculator = ap_calculator.AveragePrecisionCalculator()
    self.top_k = top_k
    self.num_examples = 0

  def accumulate(self, predictions, labels, loss):
   
    batch_size = labels.shape[0]
    mean_hit_at_one = calculate_hit_at_one(predictions, labels)
    mean_perr = calculate_precision_at_equal_recall_rate(predictions, labels)
    mean_loss = numpy.mean(loss)

    # Take the top 20 predictions.
    sparse_predictions, sparse_labels, num_positives = top_k_by_class(predictions, labels, self.top_k)
    self.map_calculator.accumulate(sparse_predictions, sparse_labels, num_positives)
    self.global_ap_calculator.accumulate(flatten(sparse_predictions), flatten(sparse_labels), sum(num_positives))

    self.num_examples += batch_size
    self.sum_hit_at_one += mean_hit_at_one * batch_size
    self.sum_perr += mean_perr * batch_size
    self.sum_loss += mean_loss * batch_size

    return {"hit_at_one": mean_hit_at_one, "perr": mean_perr, "loss": mean_loss}

  def get(self):
  
    if self.num_examples <= 0:
      raise ValueError("total_sample must be positive.")
    avg_hit_at_one = self.sum_hit_at_one / self.num_examples
    avg_perr = self.sum_perr / self.num_examples
    avg_loss = self.sum_loss / self.num_examples

    aps = self.map_calculator.peek_map_at_n()
    gap = self.global_ap_calculator.peek_ap_at_n()

    epoch_info_dict = {}
    return {"avg_hit_at_one": avg_hit_at_one, "avg_perr": avg_perr,
            "avg_loss": avg_loss, "aps": aps, "gap": gap}

  def clear(self):
    
    self.sum_hit_at_one = 0.0
    self.sum_perr = 0.0
    self.sum_loss = 0.0
    self.map_calculator.clear()
    self.global_ap_calculator.clear()
    self.num_examples = 0
