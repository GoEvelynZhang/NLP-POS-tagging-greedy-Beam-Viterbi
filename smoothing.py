import enum
import math
import numpy as np

from collections import defaultdict
from itertools import combinations


class Smoothing(enum.Enum):
    NONE = 0
    ADD_K = 2
    LINEAR = 3
    KNESER_NEY = 4
    GOOD_TURING = 5

    def _add_k(self, counts_num, counts_den, k=1):
        log_probs = {}
        vocab_size = len(counts_num)
        for key, value in counts_num.items():
            tag = tuple(key[:-1]) if len(key) > 2 else key[1]
            try:
                prob = (value + k)/(counts_den[tag] + (k*vocab_size))
                if prob ==  0.:
                    raise ZeroDivisionError
                log_probs[key] = math.log(prob)
            except ZeroDivisionError:
                log_probs[key] = float('-inf')
        return log_probs

    def _linear_interpolation(self, counts_num, ngram):
       '''
       Based on the paper: https://arxiv.org/pdf/cs/0003055.pdf
       '''
       def get_conditional_prob(tag, counts):
           l = len(tag)
           tag = tuple(np.sort(list(tag)))
           numerator = counts[l][tag]
           denominator = np.sum((list(counts[l].values()))) if l == 1 else counts[l-1][tag[:-1]]
 
           return (numerator, denominator)
       
       consts = [0] * ngram
       counts = {n+1: defaultdict(int) for n in range(ngram)}

       # Calculate counts
       for tag_group, value in counts_num.items():
           if value == 0:
               continue
           combs = [tag_group[x:y] for x, y in combinations(range(len(tag_group) + 1), r=2) if x < y]
 
           for c in combs:
               c = np.sort(list(c))
               c = tuple(c)
               counts[len(c)][c] += 1

       # Find optimal lambdas
       for tag_group, value in counts_num.items():
           if value == 0:
               continue
           best_val, const_pos = -1, None
           for i in range(1, len(tag_group) + 1):
               tag = tag_group[-i:]
               (numerator, denominator) = get_conditional_prob(tag, counts)
               if denominator-1 == 0:
                   frac = 0
               else:
                   frac = (numerator - 1)/(denominator - 1)
               if frac > best_val:
                   best_val, const_pos = frac, len(tag) - 1

           consts[const_pos] += best_val

       consts = np.array(consts, dtype=float)/np.sum(consts)
       print (f"Lambda values: {consts}")
       # Calculate probabilities
       log_probs = {}
       for tag_group, value in counts_num.items():
           total_prob = 0
           for i in range(1, len(tag_group) + 1):
               tag = tag_group[-i:]
               (numerator, denominator) = get_conditional_prob(tag, counts)
               if denominator == 0:
                   total_prob = 0
               else:
                   total_prob += (consts[len(tag) - 1] * numerator/denominator)
           log_probs[tag_group] = math.log(total_prob) if total_prob != 0 else float('-inf')
       return log_probs

    def _kneser_ney(self):
        raise Exception('Not implemented!')

    def _good_turing(self):
        raise Exception('Not implemented!')

    def smooth(self, counts_num, counts_den, **kwargs):
        '''
        Arguments:
            counts_num: Counts of n-grams occuring in the numerator
            counts_den: Counts of (n-1)-grams occuring in the denominator

        Returns:
            Smoothed MLE log probabilities
        '''
        if self == Smoothing.ADD_K:
            assert "k" in kwargs, "`k` needed to compute add-k smoothing!"
            return self._add_k(counts_num, counts_den, k=kwargs["k"])
        if self == Smoothing.LINEAR:
            assert "ngram" in kwargs, "N-Grams needed to compute linear interpolation!"
            return self._linear_interpolation(counts_num, ngram=kwargs["ngram"])
        if self == Smoothing.KNESER_NEY:
            return self._kneser_ney()
        if self == Smoothing.GOOD_TURING:
            return self._good_turing()
        
        return self._add_k(counts_num, counts_den, k=0)
