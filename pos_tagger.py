import csv
import utils

import itertools as it
import numpy as np
import pandas as pd

from collections import defaultdict
from data_loader import DataLoader
from decoder import Decoder
from smoothing import Smoothing


"""
<START> and <STOP> denote the tags for the beginning and end of a sentence.
$ cannot be used as it occurs in the existing labels.
"""
START = '<START>'
STOP = '<STOP>'

""" Contains the part of speech tagger class. """
def evaluate(data_x, data_y, model, decoder, vocabulary, beam_size=None, conf_matrix=False):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions.
    """
    """ Optimal probability """
    predicted = []
    subopt_count = 0
    for sentence, pos_tags in zip(data_x, data_y):
        pred = model.inference(sentence, decoder, beam_size)
        pred_score = model.sequence_probability(sentence, pred)
        actual_score = model.sequence_probability(sentence, pos_tags)
        if pred_score < actual_score:
            # print(f'Sub-Optimal Result: {sentence} -- {pred_score} < {actual_score}')
            subopt_count += 1
        predicted.extend(pred)

    print(f'Sub-optimal results: {subopt_count}')
    print(f'Ratio of Optimal Results: {(len(data_x) - subopt_count)*1./len(data_x)}')

    """ Accuracy statistics"""
    data_y = list(it.chain(*data_y))
    print('\n-----Dev data statistics-----')
    utils.token_accuracy(data_y, predicted)
    utils.sentence_accuracy(data_y, predicted)
    utils.calculate_f1_score(data_y, predicted)

    # Unknown tokens accuracy.
    data_y_remove_start = np.delete(data_y, np.where(np.array(data_y)=='O'))
    pred_y_remove_start = np.delete(predicted, np.where(np.array(predicted)=='O'))
 
    unk_y = []
    unk_predicted = []
    for idx, token in enumerate(list(it.chain(*data_x))):
        if token not in vocabulary:
            unk_y.append(data_y_remove_start[idx])
            unk_predicted.append(pred_y_remove_start[idx])
    
    print('\n-----Unknown words statistics-----')
    print(f'# Unknown Words: {len(unk_y)}')
    utils.token_accuracy(unk_y, unk_predicted)
    utils.calculate_f1_score(unk_y, unk_predicted)
    
    # Confusion matrix.
    if conf_matrix:
        utils.display_confusion_matrix(data_x, data_y_remove_start, pred_y_remove_start, tagset=model.tagset)
    
class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        self.ngram = None
        self.tagset = None
        self.emission_prob = None
        self.transition_prob = None

    def _create_tagset(self, tags):
        self.tagset = set(it.chain(*tags))
        print(f'No of tags: {len(self.tagset)}')

    def _set_emission_probability(self, data_x, data_y):
        count_emissions = defaultdict(int)
        count_tags = dict.fromkeys(self.tagset, 0)

        for x_i, y_i in zip(data_x, data_y):
            for word, tag in zip(x_i, y_i):
                count_emissions[(word, tag)] += 1
                count_tags[tag] += 1
        self.emission_prob = Smoothing.NONE.smooth(count_emissions, count_tags)

    def _set_transition_probability(self, data_y, smooth, k=None):
        ngram_minus1 = self.ngram - 1

        # Creating a list of possible ngram tag seqs
        count_ngram = dict.fromkeys(it.product(self.tagset, repeat=self.ngram), 0)
        for n in range(1, self.ngram):
            for item in it.product(self.tagset, repeat=n):
                count_ngram[(START,)*(self.ngram - n) + item] = 0
                count_ngram[item + (STOP,)*(self.ngram - n)] = 0

        count_ngram_minus1 = defaultdict(int)

        for y in data_y:
            padded = ([START] * (self.ngram-1)) + y + ([STOP] * (self.ngram-1))
            for idx in range(len(padded) + 1 - self.ngram):
                key = tuple(padded[idx: idx + self.ngram])
                count_ngram[key] += 1

            for idx in range(len(padded) + 1 - ngram_minus1):
                key = tuple(padded[idx: idx + ngram_minus1]) if ngram_minus1 > 1 else padded[idx]
                count_ngram_minus1[key] += 1
        
        self.transition_prob = smooth.smooth(count_ngram, count_ngram_minus1, ngram=self.ngram, k=k)
    
    def train(self, data_x, data_y, ngram=3, smooth=Smoothing.ADD_K, k=1):
        """Trains the model by computing transition and emission probabilities.

        We experiment with:
            - smoothing.
            - N-gram models with varying N.
        """
        assert ngram > 1, 'Ngram should be >= 2!'
        self.ngram = ngram
        self._create_tagset(data_y)
        
        self._set_emission_probability(data_x, data_y)
        self._set_transition_probability(data_y, smooth, k=k)
        print(f'Finished training for {self.ngram}-gram!')

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        if len(sequence) != len(tags):
            tags = tags[1:]

        score = 0.
        prev_tag = (START, ) * (self.ngram - 1)
        
        for word, tag in zip(sequence, tags):
            if tag != 'O':
                emission = self.emission_prob.get((word, tag), float('-inf'))
                transition = self.transition_prob[prev_tag + (tag, )]
                score += emission + transition
                prev_tag = prev_tag[1:] + (tag, )
        
        return score

    def inference(self, sequence, decoder=Decoder.BEAM, beam_size=3):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        if decoder == Decoder.BEAM:
            return decoder.infer(sequence, self, beam_size=beam_size)
        return decoder.infer(sequence, self)

if __name__ == "__main__":
    loader = DataLoader()
    train_x, train_y = loader.fit_transform("data/train_x.csv", "data/train_y.csv")
    dev_x, dev_y = loader.transform("data/dev_x.csv", "data/dev_y.csv", keep_start_tag=True)
    test_x, _ = loader.transform("data/test_x.csv")

    print(f'Train: {len(train_x)}, Dev: {len(dev_x)}, Test:{len(test_x)}')
    vocab_words = set(loader.vocabulary)

    pos_tagger = POSTagger()
    # Choose ngram, inference type and beam_size
    n_gram, decoder, beam_size = 3, Decoder.VITERBI, 3
    smooth_fn, k = Smoothing.ADD_K, 5e-4

    pos_tagger.train(train_x, train_y, ngram=n_gram, smooth=smooth_fn, k=k)
    evaluate(dev_x, dev_y, pos_tagger, decoder, vocabulary=vocab_words, beam_size=beam_size)
         
    # Predict tags for the test set
    test_unks, test_predictions = 0, []
    for sentence in test_x:
        test_unks += sum([1 for token in sentence if token not in vocab_words])
        test_predictions.extend(pos_tagger.inference(sentence, decoder, beam_size=beam_size))
        
        
    # Write them to a file to update the leaderboard
    print(f'\n# Unknown Words (Test): {test_unks}')
    pd.DataFrame({'id': np.arange(len(test_predictions)), 'tag': test_predictions}).to_csv(
        'results/test_y.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
