from src.models import baseline, MultiTaskNN, MultiLayerPercep, FastText, FastTextMultiTask, multiclassbaseline
from src.data.dataset import Dataset
from src.evaluation.score import report_scores, report_multiclass_scores
import random
import sys
import argparse

def run_model(model_type, in_notebook=False, is_verbose=True):

    if model_type == 'LR':
        print("\nPredicting Speaker Stance - Multi Label Logistic Regression Baseline Model ")
        model = baseline.Model()

    if model_type == 'MultiClassLR':
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
        model = FastText.My_Model(is_verbose=is_verbose)

    elif model_type == 'FastTextMT':
        print("\nPredicting Speaker Stance - FastText Multi Task Model ")
        model = FastTextMultiTask.My_Model(is_verbose=is_verbose)

    else:
        raise ValueError('Unknown Model Type')

    print("Loading Data")

    if in_notebook is True:
        data = Dataset(in_notebook=True)

    else:
        data = Dataset()

    print("Training Model")
    model.train(data.train_set())

    y, y_pred = model.test(data.test_set())
    return y, y_pred

    #gold_labels_test = data.test_set()['gold_label']
    #print(report_binary_score(gold_labels_test, predictions_test))



parser = argparse.ArgumentParser(description='Run a model for Stance Classification')
parser.add_argument('model_type', type=str, help='Select Model type')
parser.add_argument('-r', '--repeats', action='store_true', help='Select if repeats are required')
args = parser.parse_args()


if __name__ == '__main__':

    if args.repeats:
        seeds = [42, 19, 1993]
        y = None
        multi_y_pred = []
        for i in range(3):
            random.seed(seeds[i])
            print('repeat {}'.format(i+1))
            y, y_pred = run_model(args.model_type)
            multi_y_pred.append(y_pred)

        print(report_scores(y, multi_y_pred, repeats=True))





    else:
        random.seed(42)
        y, y_pred = run_model(args.model_type)
        print("\nResults on Test Data")

        if args.model_type == 'MultiClassLR':
            print(report_multiclass_scores(y, y_pred))
        else:
            print(report_scores(y, y_pred))
