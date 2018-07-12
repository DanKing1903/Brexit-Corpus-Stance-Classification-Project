import pandas as pd
import csv
import pickle

Glove = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
