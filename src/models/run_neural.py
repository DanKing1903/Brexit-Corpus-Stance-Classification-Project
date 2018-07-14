from src.models import MLP_weighted_loss, MLP_multi_task_loss, LSTM
from src.data.dataset import Dataset
from src.evaluation.score import report_scores
import numpy as np
import random
import sys





def run_model(model_type):
    print("\nPredicting Speaker Stance - Baseline Model ")
    print("Loading Data")
    data = Dataset()
    if model_type == "mlp-weighted":
        model = MLP_weighted_loss.Model()

    elif model_type == "mlp-multi":
        model = MLP_multi_task_loss.Model()

    elif model_type == "lstm":
        model =LSTM.Model()

    print("Training Model")
    model.train(data.train_set())

    print("\nResults on Test Data")
    y, y_pred = model.test(data.test_set())
    print(report_scores(y, y_pred))

    #gold_labels_test = data.test_set()['gold_label']
    #print(report_binary_score(gold_labels_test, predictions_test))


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    model_type = sys.argv[1]
    print(model_type)
    run_model(model_type)
