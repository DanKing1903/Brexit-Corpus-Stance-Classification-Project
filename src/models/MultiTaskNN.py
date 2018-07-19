import numpy as np

from keras import optimizers
from keras.layers import Input, Dense, Dropout, SimpleRNN
from keras.models import Sequential, Model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from src.features.feature_transformers import Selector, LabelTransformer, SentenceFeatures, HapaxLegomera
from src.features.target_transformers import MyMultiLabelBinarizer, MultiTaskSplitter
from src.utils.class_weights import get_weights

from keras import backend as K

from sklearn.utils import class_weight


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss


def multitask_loss(y_true, y_pred):
    # Taken from https://gist.github.com/manashmndl/9d11e269f6ba04986cf844d62ead466b#file-multitask_loss-py
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))


class My_Model(object):
    '''
    Multi label classifier model
    '''

    def __init__(self, is_verbose=True):
        #self.trainset = pd.read_csv("data/raw/train_set.csv")
        #self.testset = pd.read_csv("data/raw/test_set.csv")
        self.cv = CountVectorizer(ngram_range=(0, 2))
        self.build_pipe()
        self.is_verbose = 1 if is_verbose is True else 0

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
            ('vectorize', CountVectorizer(ngram_range=(0, 2)))])

        self.feature_pipe = Pipeline([('union', FeatureUnion(transformer_list=[('features', sent_features), ('hapax', hapax), ('Ngrams', CV)])), ('Scale', StandardScaler(with_mean=False))])

        self.mlb = Pipeline([
            ('transform', LabelTransformer()),
            ('binarize', MyMultiLabelBinarizer())])

        self.target_pipe = Pipeline([
            ('mlb', self.mlb),
            ('split', MultiTaskSplitter())])

    def build_net(self, input_dim):
        print("Building Multilayer Perceptron")
        #this will be the neural net

        #input layer
        inputs = Input(shape=(input_dim,))

        x = Dense(25, activation="relu")(inputs)

        output_layers = [Dense(1, activation="sigmoid")(x) for i in range(10)]

        #Now lets build the model
        model = Model(inputs=inputs, outputs=output_layers)

        print(model.summary())
        #stochastic gradient descent optimizer
        sgd = optimizers.SGD(lr=0.001, decay = 1e-6, momentum=0.5)
        model.compile(
            optimizer=sgd,
            #loss=multitask_loss,
            loss='binary_crossentropy',
            metrics=["accuracy"])
        return model


    def train(self, trainset):

        X = self.feature_pipe.fit_transform(trainset)
        y = self.target_pipe.fit_transform(trainset)
        num_features = X.shape[1]
        self.model = self.build_net(num_features)
        self.model.fit(X, y, epochs=500, batch_size = 32, verbose=self.is_verbose)

    def test(self, testset):
        X = self.feature_pipe.transform(testset)
        y = self.mlb.transform(testset)
        print(y.shape)
        y_pred_raw = self.model.predict(X)
        y_pred = np.column_stack(y_pred_raw)  # join the raw outputs into format for sklearn scoring
        print(y_pred.shape)
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
