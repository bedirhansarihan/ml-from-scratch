import numpy as np
import pandas as pd
import sys
import os




def sigmoid(Z: np.array):
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(X: np.array):
    A = np.maximum(0, X)
    return A
