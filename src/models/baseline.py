import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from src.models.feature_transformers import Selector, LabelTransformer, MyMultiLabelBinarizer, SentenceFeatures, HapaxLegomera
from sklearn.metrics import jaccard_similarity_score, hamming_loss, f1_score, accuracy_score


class Model(object):
    '''
    Multi label classifier model
    '''

    def __init__(self):
        self.trainset = pd.read_csv("../../data/raw/train_set.csv")
        self.testset = pd.read_csv("../../data/raw/test_set.csv")
        self.cv = CountVectorizer(ngram_range=(0, 2))
        self.model = OneVsRestClassifier(LogisticRegression())
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

        self.mlb = Pipeline([
            ('lt', LabelTransformer()),
            ('mmlb', MyMultiLabelBinarizer())])

    def train(self):
        X = self.pipe.fit_transform(self.trainset)
        y = self.mlb.fit_transform(self.trainset)
        self.model.fit(X, y)

    def test(self):
        X = self.pipe.transform(self.testset)
        self.y_test = self.mlb.transform(self.testset)
        self.y_test_pred = self.model.predict(X)
        self.print_scores(self.y_test, self.y_test_pred)
        #return self.y_test_pred

    def print_scores(self, y, y_pred):
        hamm = hamming_loss(self.y_test, self.y_test_pred)
        print('\n{:25s}{:>10.3f}\n'.format('Hamming Loss:', hamm))

        classes = [
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

        print('f1 scores \n -----------')
        f1 = f1_score(y, y_pred, average=None)
        scores = zip(classes, f1)
        for sc in sorted(scores, key=lambda s: s[1], reverse=True):
            print('{:25s}{:>10.3f}'.format(sc[0].capitalize() + ':', sc[1]))


        f1_macro = f1_score(y, y_pred, average='macro')
        f1_micro = f1_score(y, y_pred, average='micro')

        print('\n{:25s}{:10.3f}'.format('Micro-f1 score:', f1_micro))
        print('{:25s}{:>10.3f}'.format('Macro-f1 score:', f1_macro))

        accuracy = accuracy_score(y, y_pred)
        print('\n{:25s}{:10.3f}'.format('Accuracy', accuracy))


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
