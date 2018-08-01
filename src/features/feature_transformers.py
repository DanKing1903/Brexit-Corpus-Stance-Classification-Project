from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from collections import defaultdict, Counter
from nltk.tokenize.nist import NISTTokenizer
import nltk
import numpy as np
import re



class WordTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.TK = Tokenizer()

    def fit(self, X, *_):
        self.TK.fit_on_texts(X)
        return self

    def transform(self, X, *_):
        return self.TK.texts_to_sequences(X)

class Selector(BaseEstimator, TransformerMixin):
    """
    Select dataframe column, can be used in pipelines
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, *_):
        return self

    def transform(self, df):
        return df[self.key]




class SentenceFeatures(BaseEstimator, TransformerMixin):
    """
    Extract sentence features in format supporting Pipelines.

    Uses the top 10 discriminating features from Simaki (2018)) paper:
    'Evaluating stance-annotated sentences from the Brexit
    Blog Corpus: A quantitative linguistic analysis'

    These are:
    1. Average word length
    2. Conjunction frequency
    3. Sentence length in words
    4. Comma frequency
    5. Full stop frequency
    6. Hapax Legomena (number of words appearing in utterance only once)
    7. Number of different words used
    8. Sentence length in characters
    9. Punctuation frequency
    10. Hapax dislegomena (number of words appearing in utterance only twice)
    """

    def __init__(self):
        self.TK = NISTTokenizer()
        self.punct = re.compile('^[^a-zA-Z0-9_]$')

    def fit(self, *_):
        return self


    def transform(self, X, *_):
        result = []
        for sent in X:
            #print(sent)
            features = defaultdict(int)
            num_words = len(sent.split())
            tokens = self.TK.tokenize(sent, lowercase=True)
            tags = nltk.pos_tag((tokens))
            features['sent length/words'] = num_words
            counts = Counter()
            for i, token in enumerate(tokens):

                if self.punct.match(token):
                    features['punctuation'] += 1
                    if token == ',':
                        features['comma'] += 1
                    if token == '.':
                        features['period'] += 1

                else:
                    if tags[i][1] == 'CC':
                        features['conjunctions'] += 1

                    num_chars = len(re.sub(r'\W', '', token))
                    features['mean word length'] += num_chars
                    features['sent length/chars'] += num_chars
                    counts.update([token])


            features['mean word length'] /= num_words
            features['hapax legomera'] = sum([1 for k, v in counts.items() if v == 1])
            features['hapax dislegomera'] = sum([1 for k, v in counts.items() if v == 2])
            #print(counts)
            features['different words'] = len(counts.keys())
            result.append(features)
            #print(features)
        return result


class HapaxLegomera(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.TK = NISTTokenizer()
        self.punct = re.compile('^[^a-zA-Z0-9_]$')

    def compile_counts(self, X, *_):
        word_counts = Counter()
        for sent in X:
            tokens = self.TK.tokenize(sent, lowercase=True)

            for i, token in enumerate(tokens):
                if not self.punct.match(token):
                    word_counts.update([token])

        return word_counts

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):
        word_counts = self.compile_counts(X)
        result = []
        for sent in X:
            features = defaultdict(int)
            tokens = self.TK.tokenize(sent, lowercase=True)
            for i, token in enumerate(tokens):
                if not self.punct.match(token):
                    if word_counts[token] == 1:
                        features['hapax_legomera'] += 1
                    elif word_counts[token] == 2:
                        features['hapax_dislegomera'] += 1
            result.append(features)
        return result
