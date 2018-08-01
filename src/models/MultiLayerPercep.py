import numpy as np

from keras import optimizers
from keras.layers import Dense, Dropout, SimpleRNN
from keras.models import Sequential

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from src.features.feature_transformers import Selector, SentenceFeatures, HapaxLegomera
from src.features.target_transformers import MyMultiLabelBinarizer, LabelTransformer
from src.utils.class_weights import get_weights

from keras import backend as K

from sklearn.utils import class_weight


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss

class Model(object):
    '''
    Multi label classifier model
    '''

    def __init__(self, is_verbose=True):
        #self.trainset = pd.read_csv("data/raw/train_set.csv")
        #self.testset = pd.read_csv("data/raw/test_set.csv")
        self.cv = CountVectorizer(ngram_range=(0, 2))
        self.build_pipe()
        self.is_verbose = 0 if is_verbose==False else 1
        print(self.is_verbose)

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

        self.pipe = Pipeline([('union', FeatureUnion(transformer_list=[('features', sent_features), ('hapax', hapax), ('Ngrams', CV)])),('Scale', StandardScaler(with_mean = False))])

        self.mlb = Pipeline([
            ('lt', LabelTransformer()),
            ('mmlb', MyMultiLabelBinarizer())])

    def build_net(self, input_dim, class_weights):
        print("Building Multilayer Perceptron")
        #this will be the neural net
        model = Sequential()
        #input layer
        model.add(Dense(25, activation="relu", input_dim=input_dim))
        model.add(Dropout(0.2))
        model.add(Dense(25, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation="sigmoid"))
        print(model.summary())
        #stochastic gradient descent optimizer
        sgd = optimizers.SGD(lr=0.1)
        model.compile(
            optimizer='adam',
            #loss=get_weighted_loss(class_weights),
            loss='binary_crossentropy',
            metrics=["accuracy"])
        return model


    def train(self, trainset, **kwargs):

        X = self.pipe.fit_transform(trainset)
        y = self.mlb.fit_transform(trainset)
        num_features = X.shape[1]
        weights = get_weights(y)
        self.model = self.build_net(num_features, class_weights=weights)

        self.model.fit(X, y, epochs=500, batch_size=32, verbose=self.is_verbose)

    def test(self, testset):
        X = self.pipe.transform(testset)
        y = self.mlb.transform(testset)
        y_pred = self.model.predict(X)
        threshold = 0.5
        for row in y_pred:
            for i, val in enumerate(row):
                if val > threshold:
                    row[i] = 1
                elif val < threshold:
                    row[i] = 0
        print(y_pred[0])
        return y, y_pred


if __name__ == '__main__':

    from src.data.dataset import Dataset
    data = Dataset()
    model = Model()
    model.train(data.train_set())
    model.test(data.test_set())
