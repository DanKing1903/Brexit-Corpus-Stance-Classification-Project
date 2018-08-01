import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from src.features.feature_transformers import Selector, SentenceFeatures, HapaxLegomera
from src.features.target_transformers import MyMultiLabelBinarizer, LabelTransformer, MultiLabelJoiner, MyLabelEncoder, MyLabelBinarizer
from sklearn.metrics import jaccard_similarity_score, hamming_loss, f1_score, accuracy_score


class Model(object):
    '''
    Multi label classifier model
    '''

    def __init__(self):
        #self.trainset = pd.read_csv("data/raw/train_set.csv")
        #self.testset = pd.read_csv("data/raw/test_set.csv")
        self.cv = CountVectorizer(ngram_range=(0, 2))
        self.model = LogisticRegression()
        self.build_pipe()

    def build_pipe(self):
        sent_features = Pipeline(
            [('select', Selector(key='Utterance')),
             ('extract', SentenceFeatures()),
             ('vectorize', DictVectorizer())])

        hapax = Pipeline([
            ('select', Selector(key='Utterance')),
            ('extract', HapaxLegomera()),
            ('vectorize', DictVectorizer())])

        CV = Pipeline([
            ('select', Selector(key='Utterance')),
            ('cv', CountVectorizer(ngram_range=(0, 2)))])

        self.pipe = Pipeline([('union', FeatureUnion(transformer_list=[('features', sent_features), ('hapax', hapax), ('Ngrams', CV)]))])

        self.label_pipe = Pipeline([
            ('lt', LabelTransformer()),
            ('MLJ', MultiLabelJoiner()),
            ('MLB', MyLabelEncoder())]
            )


    def train(self, trainset):
        X = self.pipe.fit_transform(trainset)
        y = self.label_pipe.fit_transform(trainset)
        self.model.fit(X, y)

    def test(self, testset):
        X = self.pipe.transform(testset)
        y = self.label_pipe.fit_transform(testset)
        y_pred = self.model.predict(X)
        #self.print_scores(y, y_pred)
        return y, y_pred

    def run_model(self):
        self.train()
        self.test()

    def distribution(self, which):
        if which == 'test':
            df = self.testset
        elif which == 'train':
            df = self.trainset

        labels = df.filter(['Stance category', 'second stance category', 'third', 'fourth', 'fifth'])
        labels = labels.stack()
        print(labels.value_counts(True))

    def unique_labels(self):
        pass
