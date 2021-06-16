
import itertools as it
import pandas as pd

from collections import Counter, defaultdict
from unknown_handler import UnknownHandler, WordClass

class DataLoader:
    def __init__(self):
        self.unknown_handler = UnknownHandler()
        self.vocabulary = {}
        self.suffix_model = {}

    def _load_data(self, sentence_file, tag_file, keep_start_tag=False):
        """Loads data from two files: one containing sentences and one containing tags.

        tag_file is optional, so this function can be used to load the test data.

        Suggested to split the data by the document-start symbol.
        """
        data_x, data_y = [], []

        df = pd.merge(pd.read_csv(sentence_file), pd.read_csv(tag_file),
                    on='id', how='inner') if tag_file else pd.read_csv(sentence_file)

        words, tags = [], []
        for _, row in df.iterrows():
            if row['word'] == '-DOCSTART-':
                if len(words) > 0:
                    data_x.append(words)
                    data_y.append(tags)
                words, tags = [], ['O'] if keep_start_tag else []

            else:
                words.append(row['word'])
                if tag_file:
                    tags.append(row['tag'])
        
        # Add last sentence
        if len(words) > 0:
            data_x.append(words)
            data_y.append(tags)
        return data_x, data_y

    def _train_suffix_model(self, data, min_occurence=10, min_ratio=0.4):
        count_suffix = defaultdict(lambda : defaultdict(int))

        for word, tag in data:
            for i in range (2, 7):
                if len(word) < i:
                    break
                suffix = word[-i:]
                count_suffix[suffix][tag] += 1

        for suffix, val_dict in count_suffix.items():
            total = sum(val_dict.values())
            class_ratio = max(val_dict.values())*1.0/len(val_dict)
            if total >= min_occurence and class_ratio >= min_ratio:
                self.suffix_model[suffix] = class_ratio    

    def _create_encode_rares(self, data_x, data_y, threshold=2):
        flat_x, flat_y = list(it.chain(*data_x)), list(it.chain(*data_y))
        counter =  Counter(zip(flat_x, flat_y))

        rare_words = {(word, tag) for (word, tag), count in counter.items() if count < threshold}
        self.vocabulary = {word for (word, tag), count in counter.items() if count >= threshold}
        
        print(f'Frequent Words: {len(self.vocabulary)}, Rare Words: {len(rare_words)}')
        self._train_suffix_model(rare_words)
        return self._encode_unknowns(data_x)

    
    def _encode_unknowns(self, data_x):
        for idx, sentence in enumerate(data_x):
            for jdx, word in enumerate(sentence):
                if word not in self.vocabulary:
                    data_x[idx][jdx] = self.unknown_handler.handle_with_morph(word)
                    if data_x[idx][jdx] == word:
                        if word.lower() in self.vocabulary:
                            data_x[idx][jdx] = word.lower()
                        elif word.upper() in self.vocabulary:
                            data_x[idx][jdx] = word.upper()
                        elif word.title() in self.vocabulary:
                            data_x[idx][jdx] = word.title()
                        else:
                            suffix = self.unknown_handler.find_suffix(word, self.suffix_model)
                            if suffix is not None:
                                init = 'is_init' if jdx == 0 or data_x[idx][jdx - 1] in {'.', '?', '!'} else ''
                                hyph = 'hyph' if '-' in word else 'no-hyph'
                                casing = 'lower'
                                if word.isupper():
                                    casing = 'upper'
                                elif word.istitle():
                                    casing = 'title'
                                data_x[idx][jdx] = f'_contains_suffix_{suffix}_{casing}_{init}_{hyph}'
                            elif word[-1] == 's' or word[-1] == 'S':
                                data_x[idx][jdx] = WordClass.END_S.value
                            elif word[-1] == '.':
                                data_x[idx][jdx] = WordClass.END_DOT.value
                            elif word.isupper():
                                data_x[idx][jdx] = WordClass.ALPHA_ALL_CAPS.value
                            elif word.istitle():
                                if jdx == 0 or data_x[idx][jdx - 1] in {'.', '?', '!'}:
                                    data_x[idx][jdx] = WordClass.ALPHA_TITLE_INIT.value
                                else:
                                    data_x[idx][jdx] = WordClass.ALPHA_TITLE.value
                            elif '-' in word:
                                data_x[idx][jdx] = WordClass.OTHER_WITH_HYPHEN.value
                            else:
                                data_x[idx][jdx] = WordClass.OTHER.value
        return data_x
    
    def fit_transform(self, sentence_file, tag_file):
        data_x, data_y = self._load_data(sentence_file, tag_file)
        self.vocabulary = set(it.chain(*data_x))
        print(f'Avg. Document Length: {pd.DataFrame([len(sentence) for sentence in data_x]).mean().values[0]:.4f}')
        data_x = self._create_encode_rares(data_x, data_y)
        return data_x, data_y

    def transform(self, sentence_file, tag_file=None, keep_start_tag=False):
        data_x, data_y = self._load_data(sentence_file, tag_file, keep_start_tag)
        data_x = self._encode_unknowns(data_x)
        return data_x, data_y