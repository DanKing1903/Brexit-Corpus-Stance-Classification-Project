from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer
from collections import defaultdict, Counter
from nltk.tokenize.nist import NISTTokenizer
import nltk
import numpy as np
import re
import pickle


class LabelTransformer(BaseEstimator, TransformerMixin):
    """
    Join 5 label columns into list, correct 2 observed label errors

    """

    def __init__(self, as_sets=False):
        self.as_sets = as_sets
        return None

    def fit(self, df, *_):
        return self

    def transform(self, df, *_):
        df = df.filter(['Stance category', 'second stance category', 'third', 'fourth', 'fifth'])
        # correct errors in data:
        df.replace('concession/contrarines', np.NaN, inplace=True)
        df.replace('hypotheticallity', 'hypotheticality', inplace=True)
        if self.as_sets is True:
            y = df.stack().groupby(level=0).apply(list).apply(frozenset)

        else:
            y = df.stack().groupby(level=0).apply(list)
        return y


class MultiLabelJoiner(BaseEstimator, TransformerMixin):
    """
    concatenates labels to create a multiclass problem from a multilabel dataset
    """

    def __init__(self):
        pass

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):
        result = []
        for labels in X:
            labels = sorted(labels)
            joined_label = '_'.join(labels)
            result.append(joined_label)
        return result






class MyMultiLabelBinarizer(TransformerMixin):

    """
    Wrap MultiLabelBinarizer so it can be used in pipeline.
    See https://stackoverflow.com/questions/46162855/fit-transform-takes-2-positional-arguments-but-3-were-given-with-labelbinarize
     for problem explanation.
    """

    def __init__(self, *args, **kwargs):

        self.classes = [
            'volition',
            'prediction',
            'tact/rudeness',
            'necessity',
            'hypotheticality',
            'certainty',
            'agreement/disagreement',
            'contrariety',
            'source of knowledge',
            'uncertainty']

        self.encoder = MultiLabelBinarizer(classes=self.classes, *args, **kwargs)

    def fit(self, y, *_):
        self.encoder.fit(y)
        return self

    def transform(self, y, *_):
        yt = self.encoder.transform(y)
        return yt

    def inverse_transform(self, yt):
        y = self.encoder.inverse_transform(yt)
        return y

class MyLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()
        self.classes = []

    def fit(self, y, *_):
        with open('data/interim/label_encoding_classes', 'rb') as infile:
            labels = pickle.load(infile)
        self.encoder.fit(labels)
        self.classes = self.encoder.classes_
        return self

    def transform(self, y, *_):
        yt = self.encoder.transform(y)
        return yt


class MyLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelBinarizer()

    def fit(self, y, *_):
        self.encoder.fit(y)
        return self

    def transform(self, y, *_):
        yt = self.encoder.transform(y)
        return yt


class MultiTaskSplitter(BaseEstimator, TransformerMixin):
    """
    Class to split multi label one hot encoded targets into seperate single label targets suitable for multi task learning
    """

    def __init__(self):
        pass

    def fit(self, y, *_):
        return self

    def transform(self, X, *_):
        seperated_targets = []  # this list will contain one vector of targets per class
        for i in range(X.shape[1]):
            seperated_targets.append(X[:, i])

        return seperated_targets


def join_predictions(Y):
    joined_predictions = np.stack(Y, axis=-1)
    return joined_predictions



if __name__ == '__main__':

    y =[np.random.randint(0,5,(1, 10)) for i in range(10)]
    print(y)
    yt = join_predictions(y)

    print(yt)
