"""Dataset Reader

This module contains the class(es) and functions to read the datasets.

"""
import csv
import pandas as pd
from pathlib import Path



class Dataset(object):
    """
    Utility class to easily load the datasets for training, development and testing.
    """

    def __init__(self, in_notebook = False):
        """Defines the basic properties of the dataset reader.

        Args:
            language: The language of the dataset.
            dataset_name: The name of the dataset (all files should have it).

        """

        # TODO: Maybe the paths should be passed as parameters or read from a configuration file.
        # TODO: At some point may need dev set
        project_dir = Path(__file__).resolve().parents[2]


        self._trainset_path = str(project_dir) + r"/data/raw/train_set.csv"
        self._testset_path = str(project_dir) + r"/data/raw/test_set.csv"

        self._trainset = None
        self._testset = None

    def train_set(self):
        """list. Getter method for the training set. """
        if self._trainset is None:  # loads the data to memory once and when requested.
            self._trainset = self.read_dataset(self._trainset_path)
        return self._trainset


    def test_set(self):
        """list. Getter method for the test set. """
        if self._testset is None:  # loads the data to memory once and when requested.
            self._testset = self.read_dataset(self._testset_path)
        return self._testset

    def read_dataset(self, file_path):
        """Read the dataset file.

        Args:
            file_path (str): The path of the dataset file. The file should follow the structure specified in the
                    2018 CWI Shared Task.

        Returns:
            list. A list of dictionaries that contain the information of each sentence in the dataset.

        """
        with open(file_path) as file:
            """fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']"""

            dataset = pd.read_csv(file)
            #dataset = dataset.drop_duplicates(subset='Utterance')
        return dataset
