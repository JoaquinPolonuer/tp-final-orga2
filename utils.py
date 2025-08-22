import numpy as np


def to_array(data):
    return np.array(data) if not isinstance(data, np.ndarray) else data
