# coding: utf-8
from src.models.baseline import Model
from src.data.dataset import Dataset
data = Dataset().train_set()
y = Model().mlb.fit_transform(data)
