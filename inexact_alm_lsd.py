import numpy as np
from numpy import linalg as LA
import spams
import scipy.sparse as ssp
import scipy.io
import matplotlib.pyplot as plt
from utils import *
import time


def getGraphSPAMS_all_groups(img_shape, group_size):
    if len(img_shape) != 2:
        raise "Input lengths are incorrect"

    m = img_shape[0]
    n = img_shape[1]
    a = min(group_size, m)
    b = min(group_size, n)

    numX = m - a + 1  # number of groups on x axis
    numY = n - b + 1  # number of groups on y axis
    numGroup = numX * numY  # total number of groups

    # init graph parameters
    eta_g = np.ones(numGroup, dtype=np.float64)
    groups = ssp.csc_matrix(np.zeros((numGroup, numGroup)), dtype=bool)
    groups_var = ssp.lil_matrix(np.zeros((m * n, numGroup), dtype=bool), dtype=bool)

    # define groups
    groupIdx = 0
    for j in range(numY):
        for i in range(numX):
            indMatrix = np.zeros((m, n), dtype=bool)  # mask the size of the image
            indMatrix[i:(i + a), j:(j + b)] = True
            # groupIdx = j * (numX - 1) + i
            varsIdx = np.where(indMatrix.flatten(order='F'))
            groups_var[varsIdx, groupIdx] = True
            groupIdx += 1

    graph = {'eta_g': eta_g, 'groups': groups, 'groups_var': groups_var.tocsc()}

    return graph

# http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#sec27
def prox(G_S, lambda1, graph, num_threads=1):
    regul = 'graph'
    verbose = False  # verbosity, false by default
    pos = False  # can be used with all the other regularizations
    intercept = False  # can be used with all the other regularizations

    return spams.proximalGraph(G_S, graph, False, lambda1=lambda1, numThreads=num_threads,
                               verbose=verbose, pos=pos,
                               intercept=intercept, regul=regul)


def prox_by_frame(G_S, lambda1, graphs, num_threads=1):
    regul = 'graph'
    verbose = False  # verbosity, false by default
    pos = False  # can be used with all the other regularizations
    intercept = False  # can be used with all the other regularizations

    result = np.zeros_like(G_S)
    for frame_idx in range(G_S.shape[1]):
        result[:, [frame_idx]] = spams.proximalGraph(G_S[:, [frame_idx]], graphs[frame_idx], False,
                                                     lambda1=lambda1, numThreads=num_threads,
                                                     verbose=verbose, pos=pos,
                                                     intercept=intercept, regul=regul)

    return result

# http://thoth.inrialpes.fr/people/mairal/spams/doc-python/html/doc_spams006.html#sec25
def prox_flat(G_S, lambda1, groups, num_threads=1):
    regul = 'group-lasso-linf'
    verbose = False  # verbosity, false by default
    pos = False  # can be used with all the other regularizations
    intercept = False  # can be used with all the other regularizations

    return spams.proximalFlat(G_S, False, groups=groups, lambda1=lambda1,
                               numThreads=num_threads,
                               verbose=verbose, pos=pos,
                               intercept=intercept, regul=regul)


def inexact_alm_lsd(D0, graphs=None, groups=None, delta=10):
    # make sure D is in fortran order
    if not np.isfortran(D0):
        print('D_in is not in Fortran order')
        D = np.asfortranarray(D0)
    else:
        D = D0

    # choose between graph and flat
    if graphs is None and groups is None:
        raise "one of graphs or groups must not be None"
    elif graphs is not None and groups is not None:
        raise "only one of graphs or groups must not be None"
    else:
        if graphs is not None:
            useFlat = False
            use_prox_by_frame = type(graphs) is list or isinstance(graphs, np.ndarray)
        else:
            useFlat = True

    m, n = D.shape
    d = np.min(D.shape)

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
    max_iter = 500
    iter_out = 0
    sv = 10

    while not converged:  # Algorithm line 2
        iter_out += 1

        # SOLVE FOR L
        G_L = D - S + Y / mu  # Algorithm line 4

        u, s, vh = svd_k_largest(G_L, sv)

        # soft-thresholding
        last_nonzero_sv_idx = get_last_nonzero_idx(s - 1 / mu > 0)
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

        # Algorithm line 8
        if useFlat:
            S = prox_flat(G_S,  lambda_param / mu, groups)
        else:
            S = prox_by_frame(G_S, lambda_param / mu, graphs) if use_prox_by_frame \
                else prox(G_S, lambda_param / mu, graphs)

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
        elif iter_out >= max_iter:
            print('CONVERGED')
            break

    return L, S, iter_out, converged


def subplots_samples(sources, idx, size_factor=1):
    # plot sources on the rows and idxs on the columns
    figsize = (size_factor * len(idx), size_factor * len(sources))
    fig, axes = plt.subplots(len(sources), len(idx), figsize=figsize, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    print('Plotting...')
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


def LSD(ImData0, frame_start=0, frame_end=47, downsample_ratio=4):

    if downsample_ratio == 1:
        ImData1 = ImData0
    else:
        ImData1 = resize_with_cv2(ImData0[:, :, frame_start:(frame_end + 1)], 1 / downsample_ratio)
        # ImData1 = ImData0[::downsample_ratio, ::downsample_ratio, frame_start:(frame_end + 1)]

    normalizeImage(ImData1)

    # subtract mean
    ImMean = np.mean(ImData1)
    ImData2 = ImData1 - ImMean

    original_downsampled_shape = ImData2.shape
    w, h, frames = original_downsampled_shape
    frame_size = (w, h)

    # build graph for spams.proximalGraph
    BLOCK_SIZE = 3
    graph = getGraphSPAMS_all_groups((w, h), BLOCK_SIZE)

    # reshape so that each fame is a column
    D = ImData2.reshape((np.prod(frame_size), frames), order='F')

    L, S, iterations, converged = inexact_alm_lsd(D, graphs=graph)
    
    # mask S and reshape back to 3d array
    S_mask = foreground_mask(D, L, S).reshape(original_downsampled_shape, order='F')
    L_recon = L.reshape(original_downsampled_shape, order='F') + ImMean

    return S_mask, L_recon, ImData1


def main():
    np.random.seed(0)

    # set print precision to 2 decimal points
    np.set_printoptions(precision=2)

    # import video
    # using dtype=np.float64 to allow normalizing. use np.uint8 if not needed.
    ImData0 = np.asfortranarray(scipy.io.loadmat('data/WaterSurface.mat')['ImData'], dtype=np.float64)

    S_mask, L_recon, ImData1 = LSD(ImData0, frame_start=0, frame_end=47, downsample_ratio=4)

    print('Plotting...')
    N = 6
    video_length = ImData1.shape[2]
    subplots_samples((S_mask, L_recon, ImData1), range(0, video_length, video_length // N), size_factor=2)


if __name__ == '__main__':
    print('START')
    start = time.time()
    main()
    end = time.time()
    print('DONE')
    print(f'ELAPSED TIME: {(end - start):.3f} seconds')
