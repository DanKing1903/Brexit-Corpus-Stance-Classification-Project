from src.models.baseline import Model
from src.data.dataset import Dataset
from src.evaluation.score import report_scores
import random


def run_model(in_notebook=False):
    print("\nPredicting Speaker Stance - Baseline Model ")
    print("Loading Data")
    if in_notebook is True:
        data = Dataset(in_notebook=True)

    else:
        data = Dataset()

    model = Model()
    print("Training Model")
    model.train(data.train_set())

    print("\nResults on Test Data")
    y, y_pred = model.test(data.test_set())
    print(report_scores(y, y_pred))

    #gold_labels_test = data.test_set()['gold_label']
    #print(report_binary_score(gold_labels_test, predictions_test))


if __name__ == '__main__':
    random.seed(42)
    run_model()
