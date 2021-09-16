import numpy as np
from functools import reduce
import cv2
import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import svds

def maxWithIdx(l):
    max_idx = np.argmax(l)
    max_val = l[max_idx]
    return max_val, max_idx


def multiMatmul(*matrices, order='C'):
    return reduce(lambda result, mat: np.matmul(result, mat, order=order), matrices)


def resize_with_cv2(images, ratio):
    result_size = [int(np.ceil(images.shape[i]*ratio)) for i in [0, 1]]
    T = images.shape[2]
    result = np.empty(result_size + [T])
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    for t in range(T):
        result[:, :, t] = cv2.resize(images[:, :, t], result_size[::-1], interpolation=interpolation)
    return result


def svd_reconstruct(u, s, vh, order='C'):
    return np.matmul(u * s, vh, order=order)


def should_use_svds(d, k):
    """
    d - is the total s.v in the matrix (min(n,p))
    k - # of s.v we want to compute
    """
    ratio = 0.02 if d <= 100 \
        else 0.06 if d <= 200 \
        else 0.26 if d <= 300 \
        else 0.28 if d <= 400 \
        else 0.34 if d <= 500 \
        else 0.38

    return k / d <= ratio


def svd_k_largest(G, k):
    d = np.min(G.shape)
    use_svds = should_use_svds(d, k)
    if use_svds:
        u, s, vh = svds(G, k=k)
        return u[:, ::-1], s[::-1], vh[::-1, :]  # svds return s.v in descending order
    else:
        u, s, vh = LA.svd(G, full_matrices=False)
        return u[:, :k], s[0:k], vh[:k, :]
