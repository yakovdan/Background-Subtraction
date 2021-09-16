import numpy as np
from numpy import linalg as LA
import spams
import scipy.sparse as ssp
import scipy.io
import matplotlib.pyplot as plt
from utilities import maxWithIdx, resize_with_cv2, svd_reconstruct, svd_k_largest
import time
from scipy.sparse.linalg import svds


def getGraphSPAMS(img_shape, batch_shape):
    if len(img_shape) != 2 or len(batch_shape) != 2:
        raise "Input lengths are incorrect"

    m = img_shape[0]
    n = img_shape[1]
    a = min(batch_shape[0], m)
    b = min(batch_shape[1], n)

    numX = m - a + 1  # number of groups on x axis
    numY = n - b + 1  # number of groups on y axis
    numGroup = numX * numY  # total number of groups

    # init graph parameters
    eta_g = np.ones(numGroup, dtype=np.float64)
    groups = ssp.csc_matrix(np.zeros((numGroup, numGroup)), dtype=bool)
    groups_var = ssp.lil_matrix(np.zeros((m * n, numGroup), dtype=bool), dtype=bool)

    # define groups
    for j in range(numY):
        for i in range(numX):
            indMatrix = np.zeros((m, n), dtype=bool)  # mask the size of the image
            indMatrix[i:(i + a), j:(j + b)] = True
            groupIdx = j*(numX-1) + i
            varsIdx = np.where(indMatrix.flatten(order='F'))
            groups_var[varsIdx, groupIdx] = True

    graph = {'eta_g': eta_g, 'groups': groups, 'groups_var': groups_var.tocsc()}

    return graph


# http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#sec27
def prox(G_S, lambda1, graph):
    regul = 'graph'
    num_threads = -1  # all cores (-1 by default)
    verbose = False  # verbosity, false by default
    pos = False  # can be used with all the other regularizations
    intercept = False  # can be used with all the other regularizations

    return spams.proximalGraph(G_S, graph, False, lambda1=lambda1, numThreads=num_threads,
                               verbose=verbose, pos=pos,
                               intercept=intercept, regul=regul)


def inexact_alm_lsd(D0, graph, use_svds=True):
    # make sure D is in fortran order
    if not np.isfortran(D0):
        print('D_in is not in Fortran order')
        D = np.asfortranarray(D0)
    else:
        D = D0

    m, n = D.shape
    d = np.min(D.shape)

    delta = 10
    lambda_param = (np.sqrt(np.max((m, n))) * delta) ** (-1)

    # initialize
    Y = D
    norm_two = LA.norm(Y, ord=2)
    norm_inf = LA.norm(Y, ord=np.inf) / lambda_param
    dual_norm = np.max((norm_two, norm_inf))
    Y = Y / dual_norm

    mu = 12.5 / norm_two  # can be tuned
    rho = 1.6
    tol_out = 1e-7

    # TODO: start with known background? start with first frame?
    # L = np.zeros(D.shape, order='F')
    S = np.zeros(D.shape, order='F')

    converged = False
    iter_out = 0
    sv = 10

    while not converged:  # Algorithm line 2
        iter_out += 1

        # SOLVE FOR L
        G_L = D - S + Y / mu  # Algorithm line 4

        u, s, vh = svd_k_largest(G_L, sv)

        # soft-thresholding
        nonzero_elements = np.nonzero(s - 1 / mu > 0)
        last_nonzero_sv_idx = np.max(nonzero_elements) if len(nonzero_elements[0]) > 0 else -1
        svp = last_nonzero_sv_idx + 1

        # predicting the number of s.v bigger than 1/mu
        # ratio = s[:-1] / s[1:]
        # max_ratio, max_idx = maxWithIdx(ratio)
        # svn = svp if max_ratio <= 2 else min(svp, max_idx + 1)
        # sv = svn + 1 if svn < sv else min(svn + round(0.05 * d), d)

        sv = svp + 1 if svp < sv else min(svp + round(0.05 * d), d)

        L = svd_reconstruct(u[:, :svp], s[:svp] - 1 / mu, vh[:svp, :], order='F')  # Algorithm line 5

        # SOLVE FOR S
        G_S = D - L + Y / mu  # Algorithm line 7
        S = prox(G_S, lambda_param / mu, graph)  # Algorithm line 8

        # UPDATE Y, mu
        Z = D - L - S
        Y = Y + mu * Z  # Algorithm line 9
        mu = min(mu * rho, mu * 1e7)  # Algorithm line 10 (+limit max mu)

        # check error and convergence
        err = LA.norm(Z, ord='fro') / LA.norm(D, ord='fro')

        # print iteration info
        print(f'Iteration: {iter_out:3d} rank(L): {svp:2d} ||S||_0: {LA.norm(S.flat, ord=0):.2E} err: {err:.3E}')

        if err < tol_out:
            print('CONVERGED')
            converged = True

    return L, S, iter_out


def normalizeImage(image):
    """ Normalize image so that (min, max) -> (0, 1) """
    image -= np.min(image)
    image *= 1.0 / np.max(image)


def foreground_mask(S, D, L):
    S_abs = np.abs(S)
    S_back_temp = S_abs < 0.5 * np.max(S_abs)
    S_diff = np.abs(D - L) * S_back_temp
    positive_S_diff = S_diff[S_diff > 0]
    mu_s = np.mean(positive_S_diff)
    sigma_s = np.std(positive_S_diff)
    th = mu_s + 2 * sigma_s
    mask = S_abs > th
    return mask


def subplots_samples(sources, idx, size_factor=1):
    # plot sources on the rows and idxs on the columns
    figsize = (size_factor * len(idx),size_factor * len(sources))
    fig, axes = plt.subplots(len(sources), len(idx), figsize=figsize, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    for ix, iy in np.ndindex(axes.shape):
        ax = axes[ix, iy]
        ax.imshow(sources[ix][:, :, idx[iy]], cmap='gray', vmin=0.0, vmax=1.0)
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False
        )

    plt.show()


def main(L0=None):
    np.random.seed(0)

    # set print precision to 2 decimal points
    np.set_printoptions(precision=2)

    # import video
    # using dtype=np.float64 to allow normalizing. use np.uint8 if not needed.
    ImData0 = np.asfortranarray(scipy.io.loadmat('data/WaterSurface.mat')['ImData'], dtype=np.float64)

    original_shape = ImData0.shape

    # cut to selected frame range and downsample
    frame_start = 0
    frame_end = 47
    downsample_ratio = 4

    ImData1 = resize_with_cv2(ImData0[:, :, frame_start:(frame_end + 1)], 1/downsample_ratio)
    #ImData1 = ImData0[::downsample_ratio, ::downsample_ratio, frame_start:(frame_end + 1)]

    normalizeImage(ImData1)

    # subtract mean
    ImMean = np.mean(ImData1)
    ImData2 = ImData1 - ImMean

    original_downsampled_shape = ImData2.shape
    w, h, frames = original_downsampled_shape
    frame_size = (w, h)

    # build graph for spams.proximalGraph
    BLOCK_SIZE = (3, 3)
    graph = getGraphSPAMS((w, h), BLOCK_SIZE)

    # reshape so that each fame is a column
    D = ImData2.reshape((np.prod(frame_size), frames), order='F')

    L, S, iterations = inexact_alm_lsd(D, graph, L0)
    print(f'iterations: {iterations}')

    # mask S and reshape back to 3d array
    S_mask = foreground_mask(S, D, L).reshape(original_downsampled_shape, order='F')
    L_recon = L.reshape(original_downsampled_shape, order='F') + ImMean

    print('Plotting...')
    subplots_samples((S_mask, L_recon, ImData1), [0, 10, 20, 30, 40], size_factor=2)

    return L


if __name__ == '__main__':
    print('START')
    start = time.time()
    L0 = main()
    # main(L0)
    end = time.time()
    print('DONE')
    print(f'ELAPSED TIME: {(end-start):.3f} seconds')
