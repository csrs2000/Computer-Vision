import heapq
import random
import numbers

import numpy


class AveragePrecisionCalculator(object):
  """Calculate the average precision and average precision at n."""

  def __init__(self, top_n=None):
    
    if not ((isinstance(top_n, int) and top_n >= 0) or top_n is None):
      raise ValueError("top_n must be a positive integer or None.")

    self._top_n = top_n  # average precision at n
    self._total_positives = 0  # total number of positives have seen
    self._heap = []  # max heap of (prediction, actual)

  @property
  def heap_size(self):
    """Gets the heap size maintained in the class."""
    return len(self._heap)

  @property
  def num_accumulated_positives(self):
    """Gets the number of positive samples that have been accumulated."""
    return self._total_positives

  def accumulate(self, predictions, actuals, num_positives=None):
   
    if len(predictions) != len(actuals):
      raise ValueError("the shape of predictions and actuals does not match.")

    if not num_positives is None:
      if not isinstance(num_positives, numbers.Number) or num_positives < 0:
        raise ValueError("'num_positives' was provided but it wan't a nonzero number.")

    if not num_positives is None:
      self._total_positives += num_positives
    else:
      self._total_positives += numpy.size(numpy.where(actuals > 0))
    topk = self._top_n
    heap = self._heap

    for i in range(numpy.size(predictions)):
      if topk is None or len(heap) < topk:
        heapq.heappush(heap, (predictions[i], actuals[i]))
      else:
        if predictions[i] > heap[0][0]:  # heap[0] is the smallest
          heapq.heappop(heap)
          heapq.heappush(heap, (predictions[i], actuals[i]))

  def clear(self):
   
    self._heap = []
    self._total_positives = 0

  def peek_ap_at_n(self):
    
    if self.heap_size <= 0:
      return 0
    predlists = numpy.array(list(zip(*self._heap)))

    ap = self.ap_at_n(predlists[0],
                      predlists[1],
                      n=self._top_n,
                      total_num_positives=self._total_positives)
    return ap

  @staticmethod
  def ap(predictions, actuals):
   
    return AveragePrecisionCalculator.ap_at_n(predictions,
                                              actuals,
                                              n=None)

  @staticmethod
  def ap_at_n(predictions, actuals, n=20, total_num_positives=None):
  
    if len(predictions) != len(actuals):
      raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
      if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be 'None' or a positive integer."
                         " It was '%s'." % n)

    ap = 0.0

    predictions = numpy.array(predictions)
    actuals = numpy.array(actuals)

    # add a shuffler to avoid overestimating the ap
    predictions, actuals = AveragePrecisionCalculator._shuffle(predictions,
                                                               actuals)
    sortidx = sorted(
        range(len(predictions)),
        key=lambda k: predictions[k],
        reverse=True)

    if total_num_positives is None:
      numpos = numpy.size(numpy.where(actuals > 0))
    else:
      numpos = total_num_positives

    if numpos == 0:
      return 0

    if n is not None:
      numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
      r = min(r, n)
    for i in range(r):
      if actuals[sortidx[i]] > 0:
        poscount += 1
        ap += poscount / (i + 1) * delta_recall
    return ap

  @staticmethod
  def _shuffle(predictions, actuals):
    random.seed(0)
    suffidx = random.sample(range(len(predictions)), len(predictions))
    predictions = predictions[suffidx]
    actuals = actuals[suffidx]
    return predictions, actuals

  @staticmethod
  def _zero_one_normalize(predictions, epsilon=1e-7):
    
    denominator = numpy.max(predictions) - numpy.min(predictions)
    ret = (predictions - numpy.min(predictions)) / numpy.max(denominator,
                                                             epsilon)
    return ret
