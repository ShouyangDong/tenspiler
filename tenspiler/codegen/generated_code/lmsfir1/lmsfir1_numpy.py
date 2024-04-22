####### import statements ########
import numpy as np


def lmsfir1_np(NTAPS, input, coefficient):
    return np.sum((input[:NTAPS]) * (coefficient[:NTAPS]))


def lmsfir1_np_glued(NTAPS, input, coefficient):
    input = np.array(input).astype(np.int32)
    coefficient = np.array(coefficient).astype(np.int32)
    return lmsfir1_np(NTAPS, input, coefficient)
