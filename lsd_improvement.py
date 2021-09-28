import argparse

import numpy as np
from numpy import linalg as LA
import scipy.sparse as ssp
import scipy.io
import matplotlib.pyplot as plt
import math

import utils
from inexact_alm_lsd import subplots_samples, inexact_alm_lsd, getGraphSPAMS_all_groups, LSD, prox_by_frame, BLOCK_SIZE
from utils import *
import time

from skimage.morphology import (erosion, dilation, opening, closing)
from skimage.morphology import (rectangle, diamond, disk)

from joblib import Parallel, delayed


def get_proximal_flat_groups_nonoverlap(img_shape, batch_shape):
    if len(img_shape) != 2 or len(batch_shape) != 2:
        raise Exception("Input lengths are incorrect")

    m = img_shape[0]
    n = img_shape[1]
    a = min(batch_shape[0], m)
    b = min(batch_shape[1], n)

    # init graph parameters
    groups = np.zeros((m, n), dtype=np.int32, order='F')

    # define groups
    # idx=0 means no group on that pixel
    groupIdx = 1
    for j in range(0, n, b):
        for i in range(0, m, a):
            groups[i:(i + a), j:(j + b)] = groupIdx
            groupIdx += 1

    return groups.flatten(order='F')


def get_proximal_graph_nonoverlap(img_shape, batch_shape):
    if len(img_shape) != 2 or len(batch_shape) != 2:
        raise Exception("Input lengths are incorrect")

    m = img_shape[0]
    n = img_shape[1]
    a = min(batch_shape[0], m)
    b = min(batch_shape[1], n)

    numX = m // a  # number of groups on x axis
    numY = n // b  # number of groups on y axis
    numGroup = numX * numY  # total number of groups

    # init graph parameters
    eta_g = np.ones(numGroup, dtype=np.float64)
    groups = ssp.csc_matrix((numGroup, numGroup), dtype=bool)

    indptr = [0] * (numGroup + 1)  # number of elements in each col
    indices = []

    # define groups
    groupIdx = 0
    for j in range(0, n, b):
        for i in range(0, m, a):
            varsIdx = get_vars_idx_top_left(i, j, (a, b), img_shape)
            indptr[groupIdx + 1] = indptr[groupIdx] + len(varsIdx)
            indices.extend(varsIdx)
            groupIdx += 1

    data = np.full(len(indices), True)
    groups_var = ssp.csc_matrix((data, indices, indptr), shape=(m * n, numGroup), dtype=bool)

    graph = {'eta_g': eta_g, 'groups': groups, 'groups_var': groups_var}

    return graph


def get_proximal_graph_group_centers(img_shape, group_size, group_centers):
    """
    img_shape = (width, height) of image
    group_shape = (width, height) of groups - odd when using group_centers
    group_centers = 2D weights matrix of centers locations, or None to use all groups
    """

    # if group_centers is None:
    #     return getGraphSPAMS_all_groups(img_shape, (group_size, group_size))

    m = img_shape[0]
    n = img_shape[1]

    numGroup = np.sum(group_centers > 0)

    # init graph parameters
    eta_g = np.zeros(numGroup, dtype=np.float64)
    groups = ssp.csc_matrix((numGroup, numGroup), dtype=bool)

    indptr = [0] * (numGroup + 1)  # number of elements in each col
    indices = []

    centers_cols, centers_rows = np.where(group_centers.T > 0)  # transpose to run over cols

    group_radius = group_size
    group_width = group_radius * 2 + 1

    # define groups
    # (i,j) is the center pixel of each group
    for groupIdx in range(numGroup):
        i = centers_rows[groupIdx]
        j = centers_cols[groupIdx]

        # set group weight
        eta_g[groupIdx] = group_centers[i, j]

        # set group mask
        varsIdx = get_vars_idx_center(i, j, group_radius, img_shape)
        indptr[groupIdx + 1] = indptr[groupIdx] + len(varsIdx)
        indices.extend(varsIdx)

    data = np.full(len(indices), True)
    groups_var = ssp.csc_matrix((data, indices, indptr), shape=(m * n, numGroup), dtype=bool)

    graph = {'eta_g': eta_g, 'groups': groups, 'groups_var': groups_var}

    return graph


def inexact_alm_rpca(D0, delta=1.0, use_sv_prediction=False):
    # make sure D is in fortran order
    if not np.isfortran(D0):
        print('D_in is not in Fortran order')
        D = np.asfortranarray(D0)
    else:
        D = D0

    m, n = D.shape
    d = np.min(D.shape)

    lambda_param = (np.sqrt(np.max((m, n))) * delta) ** (-1)

    # initialize
    Y = D
    norm_two = LA.norm(Y, ord=2)
    norm_inf = LA.norm(Y, ord=np.inf) / lambda_param
    dual_norm = np.max((norm_two, norm_inf))
    Y = Y / dual_norm

    mu = 1.25 / norm_two  # can be tuned
    rho = 1.2
    tol_out = 1e-7

    # TODO: start with known background? start with first frame?
    L = np.zeros(D.shape, order='F')
    S = np.zeros(D.shape, order='F')

    converged = False
    max_iter = 500
    iter_out = 0
    sv = 10 if use_sv_prediction else d  # sv0

    while not converged:  # Algorithm line 2
        iter_out += 1

        # SOLVE FOR L
        G_L = D - S + Y / mu  # Algorithm line 4

        u, s, vh = svd_k_largest(G_L, sv)

        # soft-thresholding
        last_nonzero_sv_idx = get_last_nonzero_idx(s - 1 / mu > 0)
        svp = last_nonzero_sv_idx + 1

        # predicting the number of s.v bigger than 1/mu
        if use_sv_prediction:
            sv = svp + 1 if svp < sv else min(svp + round(0.05 * d), d)

        L = svd_reconstruct(u[:, :svp], s[:svp] - 1 / mu, vh[:svp, :], order='F')  # Algorithm line 5

        # SOLVE FOR S
        G_S = D - L + Y / mu  # Algorithm line 7
        S = np.maximum(G_S - lambda_param / mu, 0) + np.minimum(G_S + lambda_param / mu, 0)

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


def apply_background_shrinkage_operator(G, output, epsilon, background_masks):
    """
    G[pixel, frame] - input matrix
    background_masks[frame, mask_pixel] - binary column masks in a list
    epsilon = lambda / mu
    """

    for frame_idx in range(len(background_masks)):  # run over frames
        mask = background_masks[frame_idx]

        G_background = G[mask, frame_idx]
        output[mask, frame_idx] = np.maximum(1 - epsilon / LA.norm(G_background, ord=2), 0) * G_background

    return output


def inexact_alm_lsd_with_background(D0, graphs, background_masks, delta=10):
    # make sure D is in fortran order
    if not np.isfortran(D0):
        print('D_in is not in Fortran order')
        D = np.asfortranarray(D0)
    else:
        D = D0

    if type(graphs) is not list and not isinstance(graphs, np.ndarray):
        raise Exception('graphs must be list/array')

    m, n = D.shape
    d = np.min(D.shape)

    lambda_param = (np.sqrt(np.max((m, n))) * delta) ** (-1)
    background_lambda = 1e2 * lambda_param

    # initialize
    norm_two = LA.norm(D, ord=2)
    norm_inf = LA.norm(D, ord=np.inf) / lambda_param
    dual_norm = np.max((norm_two, norm_inf))
    Y = D / dual_norm

    mu = 12.5 / norm_two  # can be tuned
    rho = 1.6
    tol_out = 1e-7

    # L = np.zeros(D.shape, order='F')
    S = np.zeros(D.shape, order='F')

    converged = False
    max_iter = 500
    iter_out = 0
    sv = 10

    while not converged:  # Algorithm line 2
        iter_out += 1
        t0 = time.time()
        print(f"starting iteration :{iter_out}")
        print(f"Solving for L")
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
        print(f"Solving for S")
        G_S = D - L + Y / mu  # Algorithm line 7

        # Algorithm line 8
        print(f"prox by frame")
        S = prox_by_frame(G_S, lambda_param / mu, graphs)
        print(f"apply shrinkage ops")
        apply_background_shrinkage_operator(G_S, S, background_lambda / mu, background_masks)

        # UPDATE Y, mu
        Z = D - L - S
        Y = Y + mu * Z  # Algorithm line 9
        mu = min(mu * rho, mu * 1e7)  # Algorithm line 10 (+limit max mu)

        # check error and convergence
        err = LA.norm(Z, ord='fro') / LA.norm(D, ord='fro')

        # print iteration info
        t1 = time.time()
        print(f'Iteration: {iter_out:3d} rank(L): {svp:2d} ||S||_0: {LA.norm(S.flat, ord=0):.2E} err: {err:.3E}')
        print(f"time: {t1 - t0:.2f}s")

        if err < tol_out:
            print('CONVERGED')
            converged = True
        elif iter_out >= max_iter:
            print('NOT CONVERGED')
            break

    return L, S, iter_out, converged


def get_footprint(name, size):
    size = int(math.ceil(size))

    if name == 'rectangle':
        footprint = rectangle(size, size)
    elif name == 'disk':
        footprint = disk(size // 2)
    elif name == 'diamond':
        footprint = diamond(size // 2)
    else:
        raise Exception('unkown footprint name')

    # expand to time dimension and return
    return np.expand_dims(footprint, axis=2)


def apply_morph_ops(input, footprint_name='disk', percetage=0.05):
    im_height = input.shape[0]

    footprint_dilation = get_footprint(footprint_name, percetage * im_height)
    footprint_closing = get_footprint(footprint_name, percetage * im_height)
    print('footprint_dilation: ' + str(footprint_dilation.shape))
    print('footprint_closing: ' + str(footprint_closing.shape))

    output = input.copy(order='F')
    output = dilation(output, footprint_dilation)
    output = closing(output, footprint_closing)

    return output


def merge_masks(masks, weights, background_marker=-1):
    """
    create a weight map using a mask for each weight
    lower weight => more 
    """
    if len(masks) != len(weights):
        raise Exception('length of weights and masks must be equal')

    merged = np.ones(masks[0].shape) * background_marker

    for i in range(len(masks) - 1, -1, -1):
        merged[masks[i]] = weights[i]

    return merged


def calc_mask_percent(mask):
    return np.sum(mask > 0) / mask.size


# using different functions for timing purposes
def inexact_alm_lsd_graph(D, graph, delta):
    print("Graph")
    return inexact_alm_lsd(D, graphs=graph, delta=delta)


def inexact_alm_lsd_flat(D, groups, delta):
    print("Flat")
    return inexact_alm_lsd(D, groups=groups, delta=delta)


def build_improved_LSD_graphs(D, original_shape, weights, delta=1.0, proximal_object=None, mode=None):
    print(f'delta = {delta:f}')

    if proximal_object is None:
        L, S, iter_count, convergence = inexact_alm_rpca(D, delta=1.0)  # RPCA
    elif mode == "NONOVERLAPPING_GRAPHS":
        L, S, iter_count, convergence = inexact_alm_lsd_graph(D, proximal_object, delta)  # graphs
    elif mode == "NONOVERLAPPING_GROUPS":
        L, S, iter_count, convergence = inexact_alm_lsd_flat(D, proximal_object, delta)  # groups
    else:
        print("unknown mode")
        raise Exception("Unknown improved LSD mode")

    # mask S and reshape back to 3d array
    S_mask = foreground_mask(D, L, S, sigmas_from_mean=2) \
        .reshape(original_shape, order='F')

    # apply morphological operations

    disk_ratio = 0.05
    disk_ratio_step_size = 0.01
    total_allowed_iterations = 5
    current_iteration = 1
    max_mask_percent = 15

    S_mask_morph = apply_morph_ops(S_mask, percetage=disk_ratio) # initial guess
    weight_mask = merge_masks((S_mask, S_mask_morph), weights)
    mask_percent = calc_mask_percent(weight_mask) * 100

    print(f'mask percentage: {mask_percent:.2f}%')
    while mask_percent > max_mask_percent and current_iteration < total_allowed_iterations:
        disk_ratio -= disk_ratio_step_size
        total_allowed_iterations += 1
        S_mask_morph = apply_morph_ops(S_mask, percetage=disk_ratio) # initial guess
        weight_mask = merge_masks((S_mask, S_mask_morph), weights)
        mask_percent = calc_mask_percent(weight_mask) * 100

    print(f'final mask percentage: {mask_percent:.2f}%')
    # plot for debugging
    # normalized_weight_mask = 1 / weight_mask.copy()
    # normalizeImage(normalized_weight_mask)
    #
    # N = 6
    # video_length = original_shape[2]
    # subplots_samples([S_mask, S_mask_morph, normalized_weight_mask], range(0, video_length, video_length // N),
    #                  size_factor=4)

    print('Building graphs...')
    t0 = time.time()
    group_radius = 1
    if USE_PARALLEL:
        graphs = Parallel(n_jobs=get_usable_cores())(delayed(get_proximal_graph_group_centers) \
                                                         (weight_mask[:, :, i].shape, group_radius,
                                                          weight_mask[:, :, i]) for i in
                                                     range(weight_mask.shape[-1]))
    else:
        graphs = [get_proximal_graph_group_centers(weight_mask[:, :, i].shape,
                                                   group_radius,
                                                   group_centers=weight_mask[:, :, i])
                  for i in range(weight_mask.shape[-1])]
    t1 = time.time()
    print(f'Graphs time: {t1 - t0:.2f}s')

    background_masks = [(weight_mask[:, :, i] < 0).flatten(order='F') for i in range(weight_mask.shape[-1])]
    for mask_idx, mask in zip(range(len(background_masks)), background_masks):
        np.save(f"background_{mask_idx}", mask)
    return graphs, background_masks, iter_count, convergence


def LSD_improved(ImData0, frame_start=0, frame_end=47, downsample_ratio=1, delta=1, alg_ver=2):
    if downsample_ratio == 1:
        ImData1 = ImData0
    else:
        ImData1 = resize_with_cv2(ImData0[:, :, frame_start:(frame_end + 1)], 1 / downsample_ratio)

    normalizeImage(ImData1)

    # subtract mean
    ImMean = np.mean(ImData1)
    ImData2 = ImData1 - ImMean

    original_downsampled_shape = ImData2.shape
    w, h, frames = original_downsampled_shape
    frame_size = (w, h)

    # reshape so that each frame is a column
    D = ImData2.reshape((np.prod(frame_size), frames), order='F')

    # build graphs
    weights = (1, 1.5)
    mode = None

    if alg_ver == 2:
        proximal_object = get_proximal_flat_groups_nonoverlap(frame_size, BLOCK_SIZE)
        mode = "NONOVERLAPPING_GROUPS"
    elif alg_ver == 1:
        proximal_object = None
    else:
        print("Should not have gotten here. Something went wrong")
        raise Exception("LSD_improved wrong alg ver")
    print("Building graphs")
    graphs, background_masks, graph_iter, graph_converged = build_improved_LSD_graphs(D,
                                                                                      original_downsampled_shape,
                                                                                      weights, delta=1.0,
                                                                                      proximal_object=proximal_object,
                                                                                      mode=mode)

    print('Running LSD...')
    L, S, iterations, converged = inexact_alm_lsd_with_background(D, graphs, background_masks)

    # mask S and reshape back to 3d array
    S_mask = foreground_mask(D, L, S).reshape(original_downsampled_shape, order='F')
    L_recon = L.reshape(original_downsampled_shape, order='F')

    return S, S_mask, L_recon, ImData1, ImMean, original_downsampled_shape, \
           iterations, converged, graph_iter, graph_converged


def main(args):
    np.random.seed(0)

    # set print precision to 2 decimal points
    np.set_printoptions(precision=2)

    ImData0, _ = import_video_as_frames(args.input, args.frame_start, args.frame_end)
    graph_iter = None
    graph_converged = None

    if args.alg_ver == 1:
        print('IMPROVED LSD: RPCA')
        print_to_logfile(args.output + 'computelog.txt', f'IMPROVED algorithm: RPCA')
        t0 = time.time()
        S, S_mask, L, ImData1, ImMean, original_downsampled_shape, iterations, converged, graph_iter, graph_converged \
            = LSD_improved(ImData0, frame_start=args.frame_start,
                           frame_end=args.frame_end,
                           downsample_ratio=args.downscale,
                           alg_ver=args.alg_ver)
        t1 = time.time()
    elif args.alg_ver == 2:
        print('IMPROVED LSD: GROUP SPARSE')
        print_to_logfile(args.output + 'computelog.txt', f'IMPROVED algorithm: GROUP SPARSE')
        t0 = time.time()
        S, S_mask, L, ImData1, ImMean, original_downsampled_shape, iterations, converged, graph_iter, graph_converged \
            = LSD_improved(ImData0, frame_start=args.frame_start,
                           frame_end=args.frame_end,
                           downsample_ratio=args.downscale,
                           alg_ver=args.alg_ver)
        t1 = time.time()
    elif args.alg_ver == 0:
        print('ORIGINAL')
        print_to_logfile(args.output + 'computelog.txt', f'ORIGINAL algorithm')
        t0 = time.time()
        S, S_mask, L, ImData1, ImMean, original_downsampled_shape, iterations, converged = LSD(ImData0,
                                                                                               frame_start=args.frame_start,
                                                                                               frame_end=args.frame_end,
                                                                                               downsample_ratio=args.downscale)

        t1 = time.time()
    else:
        print('invalid algo version selected')
        print_to_logfile(args.output + 'computelog.txt', f'wrong algo version selected')
        raise Exception("wrong algo version")
    print(f'Run times: {t1 - t0:.2f}s')
    print_to_logfile(args.output + 'computelog.txt', f'Run times: {t1 - t0:.2f}s')

    np.save(args.output + "sparse", S)
    np.save(args.output + "sparse.bin", S_mask)
    np.save(args.output + "lowrank", L)
    np.save(args.output + "data", ImData1)
    with open(args.output + 'numerical_values.txt', 'w') as num_vals:
        num_vals.write(f"ImMean: {ImMean}, original downsampled shape: {original_downsampled_shape}\n")
        num_vals.write(f"iterations: {iterations}, converged: {converged}\n")
        if graph_iter is not None:
            num_vals.write(f"graph iterations: {graph_iter},graph convergence: {graph_converged}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run LSD')
    parser.add_argument('--input', type=str, default=".", help='path to input folder with jpg frames')
    parser.add_argument('--output', type=str, default=".", help='path to output folder to store binary results')
    parser.add_argument('--frame_start', type=int, default=0, help='start frame index')
    parser.add_argument('--frame_end', type=int, default=2000, help='end frame index, inclusive')
    parser.add_argument('--downscale', type=int, default=1, help='downscale factor')
    parser.add_argument('--plot', type=bool, default=False, help='plot or not')
    parser.add_argument('--alg_ver', type=int, default=0, help='algo version. 0, 1, 2')
    parser.add_argument('--parallel', type=bool, default=False, help='run in parallel mode')
    args = parser.parse_args()
    USE_PARALLEL = args.parallel
    print('START')

    enabled_str = '*ENABLED* :)' if USE_PARALLEL else 'DISABLED :('
    print(f'CORES: {num_cores}. Parallel processing {enabled_str}\n')
    args.num_cores = num_cores
    write_log_to_file(args.output + 'computelog.txt', args)
    start = time.time()
    main(args)
    end = time.time()
    print('DONE')
    print(f'ELAPSED TIME: {(end - start):.3f} seconds')
    print_to_logfile(args.output + 'computelog.txt', f'ELAPSED TIME: {(end - start):.3f} seconds')
