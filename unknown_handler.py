import enum
import numpy as np

class WordClass(enum.Enum):
    DIGIT_COLON = '_contains_digit_colon'
    DIGIT_DASH = '_contains_digit_dash'
    DIGIT_COMMA = '_contains_digit_comma'
    DIGIT_DIV = '_contains_digit_div'
    DIGIT_ONLY = '_contains_digit_only'
    DIGIT_ALPHA = '_contains_digit_alpha'
    DIGIT_ALPHA_DOT = '_contains_digit_alpha_dot'
    DIGIT_ALPHA_DASH = '_contains_digit_alpha_dash'
    DIGIT_ALPHA_DIV = '_contains_digit_alpha_div'
    DIGIT_ALPHA_APOS = '_contains_digit_alpha_apos'
    WORD_NUM_UNIT = '_contains_number_in_words_or_units'
    SINGLE_CAPS = '_contains_single_caps'
    SINGLE_LOWER = '_contains_single_lower'
    INITIALS = '_contains_initials'
    ABBR = '_contains_abbreviation'
    ALPHA_DIV = '_contains_alpha_div'
    ALPHA_TITLE_INIT = '_contains_alpha_title_case_at_start'
    ALPHA_TITLE = '_contains_alpha_title_case'
    ALPHA_LOWER = '_contains_alpha_low'
    ALPHA_ALL_CAPS = '_contains_all_caps'
    END_S = '_contains_alpha_ends_s'
    END_DOT = '_contains_dot_at_end'
    PERCENTAGE = '_contains_percent_symbol'
    OTHER_WITH_HYPHEN = '_contains_unks_with_hyphen'
    OTHER = '_contains_unks'

class UnknownHandler:
    def __init__(self):
        self.unk_model = {}

        self.units = {'mile', 'meters', 'gram', 'inch', 'year', 'hour', 'hours', 'dollar', 'foot', 'feet'}
        self.numbers = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                        'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
                        'ten', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
                        'hundred', 'thousand', 'million', 'billion'}

    def _is_only_number(self, word):
        try:
            if word.lower() in self.numbers:
                return True
            float(word)
            return True
        except ValueError:
            return False
    
    def _split_check_number(self, word, split_char):
        return all([self._is_only_number(w) for w in word.strip().split(split_char)])
    
    def _split_check_alpha(self, word, split_char):
        return split_char in word and all([w.isalpha() for w in word.strip().split(split_char)])

    def _split_check_alnum(self, word, split_char):
        return all([w.isalnum() or self.is_number(w)[0] for w in word.strip().split(split_char)])

    def _check_initials(self, word, split_char):
        return all([len(w) <= 2 and w.isupper() for w in list(filter(None, word.strip().split(split_char)))])

    def is_number(self, word):
        if self._is_only_number(word):
            return True, WordClass.DIGIT_ONLY
        if self._split_check_number(word, ':'):
            return True, WordClass.DIGIT_COLON
        if self._split_check_number(word, '-'):
            return True, WordClass.DIGIT_DASH
        if self._split_check_number(word, ','):
            return True, WordClass.DIGIT_COMMA
        if self._split_check_number(word, '\\/'):
            return True, WordClass.DIGIT_DIV
        return False, None

    def is_alnum(self, word):
        if not any(ch.isdigit() for ch in word):
            return False, None
        if word.isalnum():
            return True, WordClass.DIGIT_ALPHA
        if self._split_check_alnum(word, '-'):
            return True, WordClass.DIGIT_ALPHA_DASH
        if self._split_check_alnum(word, '.'):
            return True, WordClass.DIGIT_ALPHA_DOT
        if self._split_check_alnum(word, '\\/'):
            return True, WordClass.DIGIT_ALPHA_DIV
        if word.startswith("\'") and word[1:].isalnum():
            return True, WordClass.DIGIT_ALPHA_APOS
        return False, None

    def is_alpha(self, word):
        if self._check_initials(word, '.') or self._check_initials(word, '&') or self._check_initials(word, '-'):
            return True, WordClass.INITIALS
        if self._split_check_alpha(word, '\\/'):
            return True, WordClass.ALPHA_DIV
        if word.isalpha():
            if len(word) == 1:
                return True, WordClass.SINGLE_CAPS if word.isupper() else WordClass.SINGLE_LOWER
            if len(word) <= 4 and word.isupper():
                return True, WordClass.ABBR
        if '-' in word:
            if any([w.lower() in (self.units | self.numbers ) for w in word.split('-')]):
                return True, WordClass.WORD_NUM_UNIT
        return False, None

    def handle_with_morph(self, word):
        if '%' in word:
            return WordClass.PERCENTAGE.value

        is_number_handler = self.is_number(word)
        if is_number_handler[0]:
            return is_number_handler[1].value
        
        is_alnum_handler = self.is_alnum(word)
        if is_alnum_handler[0]:
            return is_alnum_handler[1].value

        is_alpha_handler = self.is_alpha(word)
        if is_alpha_handler[0]:
            return is_alpha_handler[1].value
        return word

    def find_suffix(self, word, suffix_model):
        best_ratio, best_suffix = 0, None
        for i in range(2, 7):
            suffix = word[-i:]
            if suffix_model.get(suffix, -1) > best_ratio:
                best_ratio, best_suffix = suffix_model[suffix], suffix
        return best_suffix
