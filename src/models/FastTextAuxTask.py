import numpy as np

from keras import optimizers
from keras.preprocessing import sequence
from keras.layers import Input, Dense, Dropout, Embedding, GlobalAveragePooling1D
from keras.models import Sequential, Model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from src.features.feature_transformers import WordTokenizer, Selector
from src.features.target_transformers import MyMultiLabelBinarizer, MultiTaskSplitter, LabelTransformer
from src.utils.metrics import Metrics

from keras import backend as K


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
        self.maxfeats = None
        self.maxlen = 40
        self.embedding_dim = 100

    def build_pipe(self):

        self.feature_pipe = Pipeline([
            ('select', Selector(key='Utterance')),
            ('tokenise', WordTokenizer())])

        self.mlb = Pipeline([
            ('transform', LabelTransformer()),
            ('binarize', MyMultiLabelBinarizer())])

        self.target_pipe = Pipeline([
            ('mlb', self.mlb),
            ('split', MultiTaskSplitter())])

        self.auxilliary pipe =  Pipeline([
            ('lt', LabelTransformer()),
            ('MLJ', MultiLabelJoiner()),
            ('MLB', MyLabelEncoder())])

    def build_net(self):
        print("Building FastText Model")
        #this will be the neural net
        #input layer
        input_layer = Input(shape=(self.maxlen,))
        embeds = Embedding(self.maxfeats, self.embedding_dim, input_length=self.maxlen)(input_layer)
        globav = GlobalAveragePooling1D()(embeds)
        output_layers = [Dense(1, activation="sigmoid")(globav) for i in range(10)]
        model = Model(inputs=input_layer, outputs=output_layers)
        print(model.summary())
        #stochastic gradient descent optimizer
        sgd = optimizers.SGD(lr=0.001, decay = 1e-6, momentum=0.5)

        model.compile(
            optimizer='adam',
            #loss=multitask_loss,
            loss = 'binary_crossentropy',
            #loss=get_weighted_loss(class_weights),
            metrics=["accuracy"])
        return model


    def train(self, trainset):
        # get word sequences
        X = self.feature_pipe.fit_transform(trainset)
        #pad sequences
        X = sequence.pad_sequences(X, maxlen=self.maxlen)
        #get targets
        y = self.target_pipe.fit_transform(trainset)
        #get vocab size from tokeniser
        self.maxfeats = len(self.feature_pipe.named_steps['tokenise'].TK.word_index)+1
        print("vocab size {}, sequence length {}".format(self.maxfeats, self.maxlen))
        self.model = self.build_net()
        self.model.fit(X, y, epochs=500, batch_size=50, verbose=self.is_verbose)

    def test(self, testset):
        X = self.feature_pipe.transform(testset)
        X = sequence.pad_sequences(X, maxlen=self.maxlen)
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