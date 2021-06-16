import enum
import numpy as np

"""
<START> and <STOP> denote the tags for the beginning and end of a sentence.
$ cannot be used as it occurs in the existing labels.
"""
START = '<START>'
STOP = '<STOP>'

class Node:
    def __init__(self, score, ptr):
        self.score = score
        self.ptr = ptr

    def __str__(self):
        return f'Score: {self.score}, Pointer: {self.ptr}'

    def __lt__(self, other):
        return self.score < other.score
    
    def __eq__(self, other):
        return self.score == other.score

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return self.score > other.score

class Decoder(enum.Enum):
    MOST_COMMON = 0
    GREEDY = 1
    BEAM = 2
    VITERBI = 3

    def _backtrack(self, sentence, trellis):
        # Backtracking the trellis to find the best path
        selected_tags, idx = [], len(sentence)
        last_tag = max(trellis[idx], key=trellis[idx].get)
        while idx > 0:
            selected_tags.append(last_tag[-1])
            last_tag = trellis[idx][last_tag].ptr
            idx -= 1
        selected_tags.append('O')
        return selected_tags[::-1]

    def _run_most_common_baseline(self, sentence, pos_tagger):
        selected_tags = ['O']
        for word in sentence:
            best_tag, best_score = 'NN', float('-inf')
            for tag in pos_tagger.tagset:
                if (word, tag) not in pos_tagger.emission_prob:
                    continue
                emission = pos_tagger.emission_prob[(word, tag)]
                if emission > best_score:
                    best_tag, best_score = tag, emission
            selected_tags.append(best_tag)
        return selected_tags

    def _run_beam_search(self, sentence, pos_tagger, beam_size):
        start_tag = (START, )*(pos_tagger.ngram - 1)
        trellis = {
            0: {start_tag: Node(0, None)}
        }

        for idx, word in enumerate(sentence):
            scores, tags = [], []
            for prev_tag in trellis[idx].keys():
                for tag in pos_tagger.tagset:
                    transition = pos_tagger.transition_prob[prev_tag + (tag, )]
                    if (word, tag) not in pos_tagger.emission_prob or transition == float('-inf'):
                        continue
                    emission = pos_tagger.emission_prob[(word, tag)]
                    score = trellis[idx][prev_tag].score + emission + transition
                    if score != float('-inf'):
                        scores.append(score)
                        tags.append((tag, prev_tag))

            if len(scores) > beam_size:
                selected = np.argpartition(scores, -beam_size)[-beam_size:]
                scores, tags = np.array(scores)[selected], np.array(tags)[selected]
            elif len(scores) == 0:
                # Fall-back in case the word is unknown
                # Default to most common tag transition
                for prev_tag, node in trellis[idx].items():
                    for tag in pos_tagger.tagset:
                        transition = pos_tagger.transition_prob[prev_tag + (tag,)]
                        scores.append(node.score + transition)
                        tags.append((tag, prev_tag))
            
            # Select best `beam_size` beams for the next iteration
            trellis[idx + 1] = {}
            for sc, (tag, prev_tag) in zip(scores, tags):
                tag = prev_tag[1:] + (tag,)
                if tag not in trellis[idx + 1] or sc > trellis[idx + 1][tag].score:
                    trellis[idx + 1][tag] = Node(sc, prev_tag)

        # Checking for STOP
        # last_tag, best_score = None, float('-inf')
        # for prev_tag in trellis[len(sentence)].keys():
        #     transition = pos_tagger.transition_prob[prev_tag + (STOP, )]
        #     score = trellis[len(sentence)][prev_tag].score + transition
        #     if score > best_score:
        #         last_tag, best_score = prev_tag, score
        return self._backtrack(sentence, trellis)

    def _run_viterbi(self, sentence, pos_tagger):
        start_tag = (START, )*(pos_tagger.ngram - 1)
        trellis = {
            0: {start_tag: Node(0, None)}
        }
        
        for idx, word in enumerate(sentence):
            trellis[idx + 1] = {}

            for tag in pos_tagger.tagset:
                if (word, tag) not in pos_tagger.emission_prob:
                    continue
                emission = pos_tagger.emission_prob[(word, tag)]
                tag = (tag, )
                for prev_tag in trellis[idx].keys():
                    transition = pos_tagger.transition_prob[prev_tag + tag]
                    if transition == float('-inf'):
                        continue
                    score = trellis[idx][prev_tag].score + emission + transition
                    tag_ngram = prev_tag[1:] + tag
                    if tag_ngram not in trellis[idx + 1] or score > trellis[idx + 1][tag_ngram].score:
                        trellis[idx + 1][tag_ngram] = Node(score, prev_tag)
            
            if len(trellis[idx + 1]) == 0:
                # Fall-back in case the word is unknown
                # Default to most common tag transition
                for prev_tag, prev_node in trellis[idx].items():
                    for tag in pos_tagger.tagset:
                        score = prev_node.score + pos_tagger.transition_prob[prev_tag + (tag,)]
                        tag_ngram = prev_tag[1:] + (tag,)
                        if tag_ngram not in trellis[idx + 1] or score > trellis[idx + 1][tag_ngram].score:
                            trellis[idx + 1][tag_ngram] = Node(score, prev_tag)
        
        return self._backtrack(sentence, trellis)

    def infer(self, sentence, pos_tagger, **kwargs):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        if self == Decoder.MOST_COMMON:
            return self._run_most_common_baseline(sentence, pos_tagger)

        if self == Decoder.VITERBI:
            return self._run_viterbi(sentence, pos_tagger)
        
        if self == Decoder.BEAM:
            assert 'beam_size' in kwargs, 'Size of beam missing in decoder!'
            return self._run_beam_search(sentence, pos_tagger, kwargs["beam_size"])
        
        # greedy search is a special case of beam_search, where beam_size=1
        return self._run_beam_search(sentence, pos_tagger, beam_size=1)