import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


def image2vector(image):
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v


def normalizeRows(x):
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = x / x_norm
    return x


def sofmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


