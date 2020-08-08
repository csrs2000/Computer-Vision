import numpy
import average_precision_calculator


class MeanAveragePrecisionCalculator(object):
  """This class is to calculate mean average precision.
  """

  def __init__(self, num_class):
    
    if not isinstance(num_class, int) or num_class <= 1:
      raise ValueError("num_class must be a positive integer.")

    self._ap_calculators = []  # member of AveragePrecisionCalculator
    self._num_class = num_class  # total number of classes
    for i in range(num_class):
      self._ap_calculators.append(
          average_precision_calculator.AveragePrecisionCalculator())

  def accumulate(self, predictions, actuals, num_positives=None):
   
    if not num_positives:
      num_positives = [None for i in predictions.shape[1]]

    calculators = self._ap_calculators
    for i in range(len(predictions)):
      calculators[i].accumulate(predictions[i], actuals[i], num_positives[i])

  def clear(self):
    for calculator in self._ap_calculators:
      calculator.clear()

  def is_empty(self):
    return ([calculator.heap_size for calculator in self._ap_calculators] ==
            [0 for _ in range(self._num_class)])

  def peek_map_at_n(self):
   
    aps = [self._ap_calculators[i].peek_ap_at_n()
           for i in range(self._num_class)]
    return aps
