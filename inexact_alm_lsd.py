import argparse
import matplotlib.pyplot as plt
import numpy as np
import spams
import scipy.io
import scipy.sparse as ssp
import time

from joblib import Parallel, delayed
from numpy import linalg as LA

import utils
from utils import *

BLOCK_SIZE = (3, 3)

def getGraphSPAMS_all_groups(img_shape, group_shape):
    if len(img_shape) != 2:
        raise Exception("Input lengths are incorrect")

    m = img_shape[0]
    n = img_shape[1]
    a = min(group_shape[0], m)
    b = min(group_shape[1], n)

    numX = m - a + 1  # number of groups on x axis
    numY = n - b + 1  # number of groups on y axis
    numGroup = numX * numY  # total number of groups

    # init graph parameters
    eta_g = np.ones(numGroup, dtype=np.float64)
    groups = ssp.csc_matrix((numGroup, numGroup), dtype=bool)

    indptr = [0] * (numGroup+1)  # number of elements in each col
    indices = []

    # define groups
    groupIdx = 0
    for j in range(numY):
        for i in range(numX):
            varsIdx = get_vars_idx_top_left(i, j, (a, b), img_shape)
            indptr[groupIdx + 1] = indptr[groupIdx] + len(varsIdx)
            indices.extend(varsIdx)
            groupIdx += 1

    data = np.full(len(indices), True)
    groups_var = ssp.csc_matrix((data, indices, indptr), shape=(m * n, numGroup), dtype=bool)

    graph = {'eta_g': eta_g, 'groups': groups, 'groups_var': groups_var}
    return graph

# http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#sec27
def prox(G_S, lambda1, graph, num_threads=utils.get_usable_cores()):
    regul = 'graph'
    verbose = False  # verbosity, false by default
    pos = False  # can be used with all the other regularizations
    intercept = False  # can be used with all the other regularizations

    return spams.proximalGraph(G_S, graph, False, lambda1=lambda1, numThreads=num_threads,
                               verbose=verbose, pos=pos,
                               intercept=intercept, regul=regul)


def prox_by_frame(G_S, lambda1, graphs):
    if USE_PARALLEL:
        frames_results = Parallel(n_jobs=get_usable_cores())(delayed(prox)(G_S[:, [frame_idx]], lambda1, graphs[frame_idx]) for frame_idx in range(G_S.shape[1]))
        return np.column_stack(frames_results)
    else:
        result = np.zeros_like(G_S)
        for frame_idx in range(G_S.shape[1]):
            result[:, [frame_idx]] = prox(G_S[:, [frame_idx]], lambda1, graphs[frame_idx], num_threads=get_usable_cores())
    return result

# http://thoth.inrialpes.fr/people/mairal/spams/doc-python/html/doc_spams006.html#sec25
def prox_flat(G_S, lambda1, groups, num_threads=utils.get_usable_cores()):
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
        raise Exception("one of graphs or groups must not be None")
    elif graphs is not None and groups is not None:
        raise Exception("only one of graphs or groups must not be None")
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
            print("Using flat")
            S = prox_flat(G_S,  lambda_param / mu, groups, num_threads=get_usable_cores())
        else:
            print("Using prox by frame")
            S = prox_by_frame(G_S, lambda_param / mu, graphs) if use_prox_by_frame \
                else prox(G_S, lambda_param / mu, graphs, num_threads=get_usable_cores())

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
            print('NOT CONVERGED')
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


def LSD(ImData0, frame_start, frame_end, downsample_ratio):

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
    graph = getGraphSPAMS_all_groups((w, h), BLOCK_SIZE)

    # reshape so that each fame is a column
    D = ImData2.reshape((np.prod(frame_size), frames), order='F')

    L, S, iterations, converged = inexact_alm_lsd(D, graphs=graph)
    
    # mask S and reshape back to 3d array
    S_mask = foreground_mask(D, L, S)
    S = S.reshape(original_downsampled_shape, order='F')
    S_mask = S_mask.reshape(original_downsampled_shape, order='F')
    L_reshaped = L.reshape(original_downsampled_shape, order='F')

    return S, S_mask, L_reshaped, ImData1, ImMean, original_downsampled_shape, iterations, converged


def main(args):
    np.random.seed(0)

    # set print precision to 2 decimal points
    np.set_printoptions(precision=2)

    # import video
    # using dtype=np.float64 to allow normalizing. use np.uint8 if not needed.

    ImData0, _ = import_video_as_frames(args.input, args.frame_start, args.frame_end)
    original_shape = ImData0.shape

    S, S_mask, L, ImData1, ImMean, original_downsampled_shape = LSD(ImData0,
                                                                    frame_start=args.frame_start,
                                                                    frame_end=args.frame_end,
                                                                    downsample_ratio=args.downscale)

    np.save(args.output+"sparse", S)
    np.save(args.output+"sparse.bin", S_mask)
    np.save(args.output+"lowrank", L)
    np.save(args.output+"data", ImData1)
    with open(args.output+'numerical_values.txt', 'w') as num_vals:
        num_vals.write(f"ImMean: {ImMean}, original downsampled shape: {original_downsampled_shape}\n")

    if args.plot:
        print('Plotting...')
        subplots_samples((S_mask, L+ImMean, ImData1), [0, 10, 20, 30, 40], size_factor=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run LSD')
    parser.add_argument('--input', type=str, default=".", help='path to input folder with jpg frames')
    parser.add_argument('--output', type=str, default=".", help='path to output folder to store binary results')
    parser.add_argument('--frame_start', type=int, default=0, help='start frame index')
    parser.add_argument('--frame_end', type=int, default=2000, help='end frame index, inclusive')
    parser.add_argument('--downscale', type=int, default=1, help='downscale factor')
    parser.add_argument('--plot', type=bool, default=False, help='plot or not')
    args = parser.parse_args()

    print('START')
    write_log_to_file(args.output+'computelog.txt', args)
    start = time.time()
    main(args)
    end = time.time()
    print('DONE')
    print(f'ELAPSED TIME: {(end - start):.3f} seconds')
    with open(args.output+'computelog.txt', 'a') as f:
        f.write(f'ELAPSED TIME: {(end - start):.3f} seconds')
