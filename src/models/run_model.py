from src.models import baseline, MultiTaskNN, MultiLayerPercep, FastText
from src.data.dataset import Dataset
from src.evaluation.score import report_scores
import random
import sys


def run_model(model_type, in_notebook=False, is_verbose=True):

    if model_type == 'baseline':
        print("\nPredicting Speaker Stance - Baseline Model ")
        model = baseline.Model()

    elif model_type == 'MTNN':
        print("\nPredicting Speaker Stance - Multi Task Model ")
        model = MultiTaskNN.My_Model(is_verbose=is_verbose)

    elif model_type == 'MLP':
        print("\nPredicting Speaker Stance - Multi Layer Perceptron Model ")
        model = MultiLayerPercep.Model(is_verbose=is_verbose)

    elif model_type == 'FastText':
        print("\nPredicting Speaker Stance - FastText Model ")
        model = FastText.My_Model(is_verbose=is_verbose)


    print("Loading Data")

    if in_notebook is True:
        data = Dataset(in_notebook=True)

    else:
        data = Dataset()

    print("Training Model")
    model.train(data.train_set())

    print("\nResults on Test Data")
    y, y_pred = model.test(data.test_set())
    print(report_scores(y, y_pred))

    #gold_labels_test = data.test_set()['gold_label']
    #print(report_binary_score(gold_labels_test, predictions_test))


if __name__ == '__main__':
    model_type = sys.argv[1]

    random.seed(42)
    run_model(model_type)
