from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def get_weights(y):
    num_classes = y.shape[1]
    weights = np.zeros([num_classes, 2])

    for i in range(num_classes):
        weights[i] = compute_class_weight('balanced', [0., 1.], y[:, i])

    return weights
