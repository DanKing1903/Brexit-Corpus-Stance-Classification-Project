from src.models import baseline, MultiTaskNN, MultiLayerPercep, FastText, FastTextMultiTask, multiclassbaseline
from src.data.dataset import Dataset
from src.evaluation.score import report_mean_scores
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from timeit import default_timer as timer
import random
import sys
import pandas as pd
import argparse


def run_model(model_type, trainset, testst, is_verbose=True,**kwargs):
    print('running model')

    if model_type == 'LR':
        print("\nPredicting Speaker Stance - Multi Label Logistic Regression Baseline Model ")
        model = baseline.Model()

    elif model_type == 'MultiClassLR':
        print("\nPredicting Speaker Stance - Multi Class Logistic Regression Baseline Model ")
        model = multiclassbaseline.Model()

    elif model_type == 'MTNN':
        print("\nPredicting Speaker Stance - Multi Task Model ")
        model = MultiTaskNN.My_Model(is_verbose=is_verbose)

    elif model_type == 'MLP':
        print("\nPredicting Speaker Stance - Multi Layer Perceptron Model ")
        model = MultiLayerPercep.Model(is_verbose=is_verbose)

    elif model_type == 'FastText':
        print("\nPredicting Speaker Stance - FastText Model ")
        model = FastText.My_Model(is_verbose=is_verbose, **kwargs)

    elif model_type == 'FastTextMT':
        print("\nPredicting Speaker Stance - FastText Multi Task Model ")
        model = FastTextMultiTask.My_Model(is_verbose=is_verbose, **kwargs)

    else:
        raise ValueError('Unknown Model Type')


    print("Training Model")
    model.train(trainset)

    y, y_pred = model.test(testset)
    return y, y_pred

    #gold_labels_test = data.test_set()['gold_label']
    #print(report_binary_score(gold_labels_test, predictions_test))



parser = argparse.ArgumentParser(description='Run a model for Stance Classification')
parser.add_argument('model_type', type=str, help='Select Model type')
args = parser.parse_args()


if __name__ == '__main__':
    seed = 42
    random.seed(seed)

    with open('data/raw/brexit_blog_corpus.csv', 'r') as file:
        df = pd.read_csv(file, encoding='utf-8')


    #X = df[['SND_studie', 'SND_dataset', 'SND_version', 'Utterance ID No', 'Utterance']]


    #y = df[['Stance category', 'second stance category', 'third', 'fourth', 'fifth']]

    embeddings = [50, 100, 150, 200, 250, 300]
    cv_scores = []
    for embedding_dim in embeddings:
        start = timer()
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
        y_accum = []

        for i, [train_idx, test_idx] in enumerate(kfold.split(df)):
            trainset = df.iloc[train_idx]
            testset = df.iloc[test_idx]
            print('Fold ', i + 1)
            kwargs = {'embedding_dim': embedding_dim}

            y, y_pred = run_model(args.model_type, trainset, testset, is_verbose=False, **kwargs)
            y_accum.append([y, y_pred])

        cv_scores.append(y_accum)
        end = timer()
        print('time elapsed: {}'.format(end - start))

    for i, embedding_dim in enumerate(embeddings):
        print('\n\nEmbedding Dimension: {}'.format(embedding_dim))
        print(report_mean_scores(cv_scores[i]))
