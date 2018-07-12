import numpy as np

from keras import optimizers
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.feature_transformers import Selector, LabelTransformer, MyMultiLabelBinarizer, SentenceFeatures, HapaxLegomera
from src.utils.class_weights import get_weights
from src.utils.vectorizers import WordIndexer

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

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

    def __init__(self):
        #self.trainset = pd.read_csv("data/raw/train_set.csv")
        #self.testset = pd.read_csv("data/raw/test_set.csv")
        self.build_pipe()

    def build_pipe(self):
        self.pipe = Pipeline([
            ('select', Selector(key='Utterance')),
            ('index', WordIndexer())])


        self.mlb = Pipeline([
            ('lt', LabelTransformer()),
            ('mmlb', MyMultiLabelBinarizer())])

    def build_net(self, input_length, class_weights):
        print("Building LSTM Nueral Network")
        #this will be the neural net

        #input layer
        model = Sequential()
        model.add(Embedding(5000, 100, input_length=input_length))
        model.add(LSTM(64))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='sigmoid'))
        print(model.summary())
        #stochastic gradient descent optimizer
        sgd = optimizers.SGD(lr=0.1)

        model.compile(
            optimizer=sgd,
            #loss=get_weighted_loss(class_weights),
            loss='binary_crossentropy',
            metrics=["accuracy"])
        return model


    def train(self, trainset):

        X = self.pipe.fit_transform(trainset)
        y = self.mlb.fit_transform(trainset)
        num_features = X.shape[1]
        weights = get_weights(y)
        self.model = self.build_net(input_length=num_features, class_weights=weights)
        self.model.fit(X, y, epochs=10, batch_size=32)

    def test(self, testset):
        X = self.pipe.transform(testset)
        y = self.mlb.transform(testset)
        y_pred = self.model.predict(X)
        for row in y_pred:
            for i, val in enumerate(row):
                if val > 0.5:
                    row[i] = 1
                elif val < 0.5:
                    row[i] = 0
        print(y_pred[0])
        return y, y_pred


if __name__ == '__main__':

    from src.data.dataset import Dataset
    data = Dataset()
    model = Model()
    model.train(data.train_set())
    model.test(data.test_set())
