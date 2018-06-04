from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from itertools import product
from string import ascii_lowercase


class TokenGenerator:
    '''
    Generator to tokenize the documents using a vectorier analyzer
    '''
    def __init__(self, docs):
        self.docs = docs
        self.analyer = TfidfVectorizer().build_analyzer()

    def __iter__(self):
        for doc in self.docs:
            tokens = [w for w in self.analyer(doc)]
            yield tokens


def get_token_pattern():
    '''
    Token pattern finds words that start and end with letter.
    Thus no numbers at the beginning or end.
    '''
    token_pattern = '(?ui)\\b[a-zA-Z]*[a-z]+\\w*\\b'
    return token_pattern


def get_stop_words():
    cusotm_stop_words = \
       ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        'll', 'am', 'pm', 'minutes', 'hours', 'time', 'reasonable',
        'day', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
        'days', 'week', 'saturday', 'sunday', 'weekday', 'weekend', 'month',
        'year',

        # For diabetes dataset
        'episode', 'unspecified', 'specified', 'stated', 'condition', 'care',
        'not', 'applicable', 'condition', 'effect', 'classified', 'personal',
        'involving', 'unknown', 'degree'
        ]

    # Remove all 2 letter words
    all_two_letter_words = list(map(''.join, product(ascii_lowercase, repeat=2)))

    added_stop_words = cusotm_stop_words + all_two_letter_words

    stop_words = text.ENGLISH_STOP_WORDS.union(added_stop_words)

    return stop_words
