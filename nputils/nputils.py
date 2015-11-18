import numpy as np


def simplex_volume(simplex_points):
    return np.abs(np.linalg.det(np.transpose(simplex_points[1:] - simplex_points[0])))


def calculate_sparsity(data):
    nonz = np.count_nonzero(data)
    return float(data.size - nonz) / data.size


def average_auc(scores):
    return np.mean(scores)


def sign(a):
    s = np.sign(a)
    s[s == 0] = 1
    return s


def msqrt(M):
    (u, s, vt) = np.linalg.svd(M)
    ms = np.dot(u, np.dot(np.diag(np.sqrt(s)), vt))
    return ms
