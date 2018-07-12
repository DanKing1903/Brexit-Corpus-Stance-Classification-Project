from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from nltk.tokenize.nist import NISTTokenizer
import numpy as np


class WordIndexer(BaseEstimator, TransformerMixin):
    """
    code modified from https://github.com/adventuresinML/adventures-in-ml-code/blob/master/keras_lstm.py
    """

    def __init__(self, reverse=False):

        self.TK = NISTTokenizer()
        self.word2idx = None
        self.sent_size = 0

    def build_vocab(self, X, *_):
        counter = Counter()
        max_len = 0
        for sent in X:
            tokens = self.TK.tokenize(sent, lowercase=True)
            if len(tokens) > max_len:
                max_len = len(tokens)
            counter.update(tokens)

        sort_by_counts = sorted(counter.items(), key=lambda x: x[1])
        words, counts = zip(*sort_by_counts)

        word2idx = dict(zip(words, range(1, len(words) + 1)))
        return word2idx, max_len


    def fit(self, X, *_):
        self.word2idx, self.sent_size = self.build_vocab(X)
        return self

    def transform(self, X, *_):
        vec = np.zeros((len(X), self.sent_size + 25))
        for i, sent in enumerate(X):
            tokens = self.TK.tokenize(sent, lowercase=True)
            for j, tok in enumerate(tokens):
                vec[i][j] = self.word2idx[tok]
            return vec


if __name__ == '__main__':
    from src.data.dataset import Dataset
    trainset = Dataset().train_set()
    WI = WordIndexer()
    test = WI.fit_transform(trainset['Utterance'])
    print(test[0])
    print(len(WI.word2idx))
    print(WI.sent_size)
