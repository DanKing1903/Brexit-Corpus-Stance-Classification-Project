import numpy as np
from src.data.dataset import Dataset

from keras.preprocessing import sequence
from keras.layers import Input, Dense, Dropout, Embedding, GlobalAveragePooling1D
from keras.models import Model
from keras.utils import np_utils
from keras.regularizers import l2

from sklearn.feature_extraction.text import CountVectorizer

from src.features import pipelines


from src.evaluation.score import report_scores, report_multiclass_scores, report_mean_scores, report_mean_multiclass_scores


class FastText(object):
    '''
    Fast Text Classifier
    '''

    def __init__(self, domain, embedding_dim=100, is_verbose=True, epochs=300, probs=False):
        #self.trainset = pd.read_csv("data/raw/train_set.csv")
        #self.testset = pd.read_csv("data/raw/test_set.csv")
        self.domain = domain
        self.probs = probs
        self.epochs = epochs
        print(domain)
        self.build_pipe()
        self.is_verbose = 1 if is_verbose is True else 0
        self.maxfeats = None
        self.maxlen = None
        self.embedding_dim = embedding_dim

    def build_pipe(self):

        self.feature_pipe = pipelines.get_feature_pipe('embeddings')

        self.label_pipe = pipelines.get_label_pipe(self.domain)

    def build_net(self):
        print("Building FastText Model")
        #this will be the neural net
        #input layer
        input_layer = Input(shape=(self.maxlen,))
        embeds = Embedding(self.maxfeats, self.embedding_dim, input_length=self.maxlen)(input_layer)
        globav = GlobalAveragePooling1D()(embeds)
        #dropout=0.5
        #globav = Dropout(dropout)(globav)

        if self.domain == 'multi-label':
            output_layer = Dense(10, activation="sigmoid")(globav)
            model = Model(inputs=input_layer, outputs=output_layer)

        elif self.domain == 'multi-class':
            self.output_size = len(list(self.label_pipe.named_steps['MLE'].classes))
            output_layer = Dense(self.output_size, activation="softmax")(globav)
            model = Model(inputs=input_layer, outputs=output_layer)

        elif self.domain == 'multi-task':
            output_layers = [Dense(1, activation="sigmoid")(globav) for i in range(10)]
            model = Model(inputs=input_layer, outputs=output_layers)

        print(model.summary())
        model.compile(
            optimizer='adam',
            #loss=multitask_loss,
            loss='binary_crossentropy',
            #loss=get_weighted_loss(class_weights),
            metrics=["accuracy"])
        return model


    def train(self, trainset):
        # get word sequences
        X = self.feature_pipe.fit_transform(trainset)
        #pad sequences
        longest_seq = max(X, key=len)
        self.maxlen = len(longest_seq)
        X = sequence.pad_sequences(X, maxlen=self.maxlen)
        #get targets
        y = self.label_pipe.fit_transform(trainset)
        if self.domain == 'multi-class':
            y = np_utils.to_categorical(y)
        #get vocab size from tokeniser
        self.maxfeats = len(self.feature_pipe.named_steps['sequence'].TK.word_index)+1
        print("vocab size {}, sequence length {}".format(self.maxfeats, self.maxlen))
        self.model = self.build_net()
        hist = self.model.fit(X, y, epochs=self.epochs, batch_size=50, verbose=self.is_verbose)
        return hist

    def test(self, testset):
        X = self.feature_pipe.transform(testset)
        X = sequence.pad_sequences(X, maxlen=self.maxlen)
        y = self.label_pipe.transform(testset)
        y_pred = self.model.predict(X)

        if self.domain == 'multi-class':
                y_pred = [np.argmax(yi) for yi in y_pred]

        elif self.domain == 'multi-label' or self.domain == 'multi-task':
            if self.domain == 'multi-task':
                print('stacking')
                y = np.column_stack(y)
                y_pred = np.column_stack(y_pred)

            if not self.probs:
                y_pred = np.where(y_pred > 0.5,1,0) #  binarize result


            """threshold = 0.5
            for row in y_pred:
                for i, val in enumerate(row):
                    if val > threshold:
                        row[i] = 1
                    elif val < threshold:
                        row[i] = 0"""


        return y, y_pred


if __name__ == '__main__':
    data = Dataset()

    print("Training Model")
    for domain in ['multi-label']:
        model = FastText(domain, is_verbose=True)
        model.train(data.train_set())

        y, y_pred = model.test(data.test_set())

        print("\nResults on Test Data")

        if domain is 'multi-class':
            print(report_multiclass_scores(y, y_pred))

        else:
            print(report_scores(y, y_pred))
