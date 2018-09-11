from src.classifiers.logisitic_regression import LR
from src.classifiers.fast_text_net import FastText
from src.classifiers.multi_layer_perceptron import MLP

from src.data.dataset import Dataset
from src.data.dataset import Dataset
from src.evaluation.score import report_scores, report_multiclass_scores, report_mean_scores, report_mean_multiclass_scores
import random
import sys
import pickle
import argparse


def run_model(model_type, domain, is_verbose=True, probs=False, get_history=False):
    if model_type == 'logistic_regression':
        model = LR(domain, probs=probs)
    elif model_type =='fast_text':
        model = FastText(domain,probs=probs, is_verbose=is_verbose)
    elif model_type == 'mlp':
        model = MLP(domain, is_verbose=is_verbose)
    else:
        raise ValueError('incorrect model type choice')

    print("Loading Data")
    data = Dataset()

    print("Training Model")
    hist = model.train(data.train_set())

    if get_history is True:
        return hist

    else:
        y, y_pred = model.test(data.test_set())
        return y, y_pred






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a model for Stance Classification')
    parser.add_argument('model_type', type=str, help='Select model type', choices=['logistic_regression', 'fast_text', 'mlp'])
    parser.add_argument('domain', type=str, help='Select classification domain', choices=['multi-class', 'multi-label', 'multi-task'])
    parser.add_argument('-s', '--save_model', action='store_true', help='Save model on completion')
    #parser.add_argument('-r', '--repeats', action='store_true', help='Select if repeats are required')
    args = parser.parse_args()

    model_type = args.model_type
    domain = args.domain
    if model_type == 'logisitic_regression' and domain == 'multi-task':
        raise ValueError('Multi-task logisitic_regression not supported')





    if args.save_model is True:
        # this options is to be completed
        raise ValueError('Save model option is not yet completed')

    random.seed(42)
    y, y_pred = run_model(model_type, domain)
    print("\nResults on Test Data")

    if domain == 'multi-class':
        print(report_multiclass_scores(y, y_pred))
    else:
        print(report_scores(y, y_pred))





    """if args.repeats:
        seeds = [42, 19, 1993]
        y_accum = []
        for i in range(3):
            random.seed(seeds[i])
            print('repeat {}'.format(i + 1))
            y, y_pred = run_model(model_type, domain)
            y_accum.append([y, y_pred])

        if domain == 'multi-class':
            print(report_mean_multiclass_scores(y_accum))

        else:
            print(report_scores(y_accum))"""
