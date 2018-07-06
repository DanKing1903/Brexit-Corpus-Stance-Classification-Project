from src.models.baseline import Model
from src.data.dataset import Dataset
from src.evaluation.score import report_scores



def run_model():
    print("\nPredicting Speaker Stance - Baseline Model ")
    print("Loading Data")
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
    run_model()
