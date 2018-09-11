import numpy as np

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import np_utils

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from src.features.feature_transformers import Selector, SentenceFeatures, HapaxLegomera
from src.features.target_transformers import MyMultiLabelBinarizer, LabelTransformer
from src.features import pipelines
from src.data.dataset import Dataset
from src.utils.class_weights import get_weights

from keras import backend as K

from sklearn.utils import class_weight
from src.evaluation.score import report_scores, report_multiclass_scores, report_mean_scores, report_mean_multiclass_scores


class MLP(object):
    '''
    Multi label classifier model
    '''

    def __init__(self, domain,epochs=500, is_verbose=True):

        self.epochs=epochs
        self.is_verbose = 0 if is_verbose is False else 1
        self.domain = domain
        print(self.is_verbose)
        self.build_pipe()

    def build_pipe(self):
        self.feature_pipe = pipelines.get_feature_pipe('engineered', scaling=True)

        self.label_pipe = pipelines.get_label_pipe(self.domain)

    def build_net(self, input_dim,):
        print("Building Multilayer Perceptron")
        #this will be the neural net

        input_layer = Input(shape=(input_dim,))
        #input layer
        x = Dense(25, activation="relu", input_dim=input_dim)(input_layer)
        x = Dropout(0.2)(x)
        x = Dense(25, activation="relu")(x)
        x = Dropout(0.2)(x)

        if self.domain == 'multi-label':
            output_layer = Dense(10, activation="sigmoid")(x)
            model = Model(inputs=input_layer, outputs=output_layer)

        elif self.domain == 'multi-class':
            self.output_size = len(list(self.label_pipe.named_steps['MLE'].classes))
            output_layer = Dense(self.output_size, activation="softmax")(x)
            model = Model(inputs=input_layer, outputs=output_layer)

        elif self.domain == 'multi-task':
            output_layers = [Dense(1, activation="sigmoid")(x) for i in range(10)]
            model = Model(inputs=input_layer, outputs=output_layers)

        print(model.summary())
        #stochastic gradient descent optimizer
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=["accuracy"])
        return model


    def train(self, trainset, **kwargs):

        X = self.feature_pipe.fit_transform(trainset)
        y = self.label_pipe.fit_transform(trainset)
        if self.domain == 'multi-class':
            y = np_utils.to_categorical(y)

        num_features = X.shape[1]
        self.model = self.build_net(num_features)

        self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=self.is_verbose)

    def test(self, testset):

        X = self.feature_pipe.transform(testset)
        y = self.label_pipe.transform(testset)
        y_pred = self.model.predict(X)

        if self.domain == 'multi-class':
                y_pred = [np.argmax(yi) for yi in y_pred]

        elif self.domain == 'multi-label' or self.domain == 'multi-task':
            if self.domain == 'multi-task':
                print('stacking')
                y = np.column_stack(y)
                y_pred = np.column_stack(y_pred)

            y_pred = np.where(y_pred > 0.5,1,0) #  binarize result
        return y, y_pred


if __name__ == '__main__':
    data = Dataset()

    print("Training Model")
    for domain in ['multi-task', 'multi-label', 'multi-class']:
        model = MLP(domain)
        model.train(data.train_set())

        y, y_pred = model.test(data.test_set())

        print("\nResults on Test Data")

        if domain is 'multi-class':
            print(report_multiclass_scores(y, y_pred))

        else:
            print(report_scores(y, y_pred))
