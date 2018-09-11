from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import random
seed = 42
random.seed(seed)

from src.classifiers.logisitic_regression import LR
from src.classifiers.fast_text_net import FastText
from src.classifiers.multi_layer_perceptron import MLP
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from src.data.dataset import Dataset
from src.data.dataset import Dataset
from src.evaluation.score import report_scores, report_multiclass_scores, report_mean_scores, report_mean_multiclass_scores
from timeit import default_timer as timer

import sys
import pickle
import argparse
import string

import pandas as pd


def run_model(model_type, domain, trainset, testset, is_verbose, **kwargs):
    if model_type =='fast_text':
        model = FastText(domain, is_verbose=is_verbose, **kwargs)

    elif model_type =='logistic_regression':
        model = LR(domain)

    elif model_type == 'mlp':
        model = MLP(domain, is_verbose=is_verbose, **kwargs)
    else:
        raise ValueError('incorrect model type choice {}'.format(model_type))

    print("Loading Data")
    data = Dataset()

    print("Training Model")
    model.train(trainset)

    y, y_pred = model.test(testset)
    return y, y_pred




def run_cv(model_types, domains, outfile=None, is_verbose=False):
    print(model_types)

    df_data = []

    with open('data/raw/brexit_blog_corpus.csv', 'r') as file:
        df = pd.read_csv(file, encoding='utf-8')
        if not type(model_types) == list:
            model_types = [model_types]
            print(model_types)

        if not type(domains) == list:
            domains = [domains]
            print(domains)
        for model_type in model_types:

            for domain in domains:
                if domain == 'multi-task' and model_type == 'logistic_regression':
                    continue

                params = [100,159,200,250,300,350,400]
                cv_scores = []
                for param in params:
                    start = timer()
                    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
                    y_accum = []

                    for i, [train_idx, test_idx] in enumerate(kfold.split(df)):
                        trainset = df.iloc[train_idx]
                        testset = df.iloc[test_idx]
                        print('Fold ', i + 1)
                        kwargs = {'epochs': param}
                        print(kwargs)


                        y, y_pred = run_model(model_type, domain, trainset, testset, is_verbose=is_verbose, **kwargs)
                        y_accum.append([y, y_pred])

                    cv_scores.append(y_accum)
                    end = timer()
                    print('time elapsed: {}'.format(end - start))

                for i, dim in enumerate(params):
                    print('\n\nEpochs: {}'.format(dim))
                    if model_type == 'fast_text':
                        model_name = 'FastText'

                    elif model_type == 'logistic_regression':
                        model_name = 'LR'
                    elif model_type == 'mlp':
                        model_name = 'MLP'
                    domain_name = string.capwords(domain)
                    row = [model_name, domain_name, dim]
                    if domain == 'multi-class':
                        print(report_mean_multiclass_scores(cv_scores[i]))
                        row.extend(report_mean_multiclass_scores(cv_scores[i], print_result=False))
                    else:
                        print(report_mean_scores(cv_scores[i]))
                        row.extend(report_mean_scores(cv_scores[i], print_result=False))
                    print(row)
                    df_data.append(row)


    cv_df = pd.DataFrame(df_data, columns=['Model', 'Domain', 'Epochs', 'F1 Macro', ' F1 Micro', 'EMR'])
    cv_df.set_index(['Model', 'Domain', 'Empochs'], inplace=True)
    embed_tuning = False
    if embed_tuning is True:
        cv_df = cv_df.unstack()
        cv_df = cv_df.swaplevel(axis=1).stack()


    cv_df = cv_df.round(3)
    print(cv_df)
    if outfile:
        cv_df.to_latex(outfile,multirow=True)
    return cv_df






if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Run a model for Stance Classification')
    parser.add_argument('model_type', type=str, help='Select model type', choices=['logistic_regression', 'fast_text', 'mlp', 'neural','all'])
    parser.add_argument('domain', type=str, help='Select classification domain', choices=['multi-class', 'multi-label', 'multi-task', 'all'])
    parser.add_argument('-s', '--save_model', action='store_true', help='Save model on completion')
    parser.add_argument('-w', '--write_latex',  help='save table to latex file', dest='file_path')
    parser.add_argument('-v', '--verbose',  help='verbose mode', action='store_true')
    #parser.add_argument('-r', '--repeats', action='store_true', help='Select if repeats are required')
    args = parser.parse_args()
    """try:
        if args.is_verbose:
            print('Running in verbose mode')

    except:
        args.verbose = False
        print('Running in quiet mode')"""
    print(args)

    if args.model_type ==  'all':
        model_type=['logistic_regression', 'fast_text', 'mlp']

    elif args.model_type == 'neural':
        model_type=['fast_text', 'mlp']

    else:
        model_type = args.model_type


    if args.domain == 'all':
        domain = ['multi-class', 'multi-label', 'multi-task']

    else:
        domain = args.domain

    if args.file_path:
        run_cv(model_type, domain, outfile=args.file_path, is_verbose=args.verbose)

    else:
        run_cv(model_type, domain, is_verbose=args.verbose)


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
