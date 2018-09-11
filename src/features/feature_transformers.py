from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from collections import defaultdict, Counter
from nltk.tokenize.nist import NISTTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import re

class Stops_Stems(BaseEstimator, TransformerMixin):
    def init(self, stemming):
        self.stemming = stemming

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):
        stop_words = set(stopwords.words('english'))
        if self.stemming:
            ps = nltk.stemmer.PorterStemmer()
        result = []
        for sent in X:
            word_tokens = word_tokenize(sent)
            filtered_sent = [w for w in word_tokens if w not in stop_words]
            if self.stemming:
                stemmed_sent = [ps.stem(w) for w in filtered_sent]
                result.append(stemmed_sent)
            else:
                result.append(filtered_sent)






class WordTokenizer2(BaseEstimator, TransformerMixin):
    def __init__(self, char_level=False, strip_punctuation=False, ngram_range=(1,1)):
        self.TK = NISTTokenizer()
        self.word_index = dict()
        self.index_word = dict()
        self.strip_punctuation = strip_punctuation
        self.punct = re.compile('^[^a-zA-Z0-9_]$')


    def fit(self, X, *_):
        i = 1
        for sent in X:
            tokens = self.TK.tokenize(sent, lowercase=True)
            for t in tokens:
                if self.strip_punctuation:
                    if not self.punct.match(t):
                        if t not in self.word_index:
                            self.word_index[t] = i
                            self.index_word[i] = t
                            i += 1

                else:
                    if t not in self.word_index:
                        self.word_index[t] = i
                        self.index_word[i] = t
                        i += 1

        return self


    def transform(self, X, *_):

        #returns sequence of form [1,2,3,4]

        sequences = []
        for sent in X:
            seq = []
            tokens = self.TK.tokenize(sent, lowercase=True)
            for t in tokens:
                if self.strip_punctuation:
                    if not self.punct.match(t):
                        if t in self.word_index:
                            seq.append(self.word_index[t])

                else:
                    if t in self.word_index:
                        seq.append(self.word_index[t])

            sequences.append(seq)

        return sequences


class WordTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.TK = Tokenizer()


    def fit(self, X, *_):
        self.TK.fit_on_texts(X)
        self.word_index = self.TK.word_index
        return self

    def transform(self, X, *_):
        return self.TK.texts_to_sequences(X)

#class AddNGrams(BaseEstimator, TransformerMixin):
    #def __init__(self, ngram_range=)


class MyCountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range):
        self.CV = CountVectorizer(ngram_range=ngram_range)

    def fit(self, X, *_):
        self.CV.fit(X)

    def transform(self, X, *_):
        wd_2_idx = self.CV.vocabulary_





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




class MyWordSequencer(BaseEstimator, VectorizerMixin):

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype

    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = lambda: 1 + vocabulary.__len__()

        analyze = self.build_analyzer()

        X = []
        for doc in raw_documents:
            feature_list = []
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    feature_list.append(feature_idx)

                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue
            X.append(feature_list)



        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")


        return vocabulary, X

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.


        self._validate_vocabulary()

        vocabulary, X = self._count_vocab(raw_documents,
                                          self.fixed_vocabulary_)

        self.vocabulary_ = vocabulary

        return X

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """

        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()

        self._check_vocabulary()

        # use the same matrix-building strategy as fit_transform
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)

        return X
