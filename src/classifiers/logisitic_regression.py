import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from src.features.feature_transformers import Selector, SentenceFeatures, HapaxLegomera
from src.features.target_transformers import MyMultiLabelBinarizer, LabelTransformer
from sklearn.metrics import jaccard_similarity_score, hamming_loss, f1_score, accuracy_score
from src.features import pipelines
from src.data.dataset import Dataset
from src.evaluation.score import report_scores, report_multiclass_scores, report_mean_scores, report_mean_multiclass_scores



class LR(object):
    '''
    Multi label classifier model
    '''

    def __init__(self, domain, probs = False):
        if domain not in set(['multi-label', 'multi-task', 'multi-class']):
            raise ValueError('Incorrect domain config: {}'.format(domain))

        if domain == 'multi-task':
            raise ValueError('Multi-task Logistic Regression not supported')

        self.domain = domain
        if self.domain == 'multi-label':
            self.model = OneVsRestClassifier(LogisticRegression())

        elif self.domain == 'multi-class':
            self.model = LogisticRegression()

        self.probs = probs
        self.build_pipe()

    def build_pipe(self):
        self.feature_pipe = pipelines.get_feature_pipe('engineered')
        self.label_pipe = pipelines.get_label_pipe(self.domain)

    def train(self, trainset):
        X = self.feature_pipe.fit_transform(trainset)
        y = self.label_pipe.fit_transform(trainset)
        self.model.fit(X, y)

    def test(self, testset):
        X = self.feature_pipe.transform(testset)
        y = self.label_pipe.transform(testset)

        if self.probs:
            y_pred = self.model.predict_proba(X)
        else:
            y_pred = self.model.predict(X)

        return y, y_pred


if __name__ == '__main__':
    data = Dataset()

    print("Training Model")
    for domain in ['multi-label', 'multi-class']:
        model = LR(domain)
        model.train(data.train_set())

        y, y_pred = model.test(data.test_set())

        print("\nResults on Test Data")

        if domain is 'multi-class':
            print(report_multiclass_scores(y, y_pred))

        else:
            print(report_scores(y, y_pred))
