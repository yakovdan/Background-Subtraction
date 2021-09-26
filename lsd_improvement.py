import numpy as np
from numpy import linalg as LA
import scipy.sparse as ssp
import scipy.io
import matplotlib.pyplot as plt
import math

from inexact_alm_lsd import subplots_samples, inexact_alm_lsd, getGraphSPAMS_all_groups, LSD, prox_by_frame
from utils import *
import time

from skimage.morphology import (erosion, dilation, opening, closing)
from skimage.morphology import (rectangle, diamond, disk)

from joblib import Parallel, delayed


def get_proximal_flat_groups_nonoverlap(img_shape, batch_shape):
    if len(img_shape) != 2 or len(batch_shape) != 2:
        raise "Input lengths are incorrect"

    m = img_shape[0]
    n = img_shape[1]
    a = min(batch_shape[0], m)
    b = min(batch_shape[1], n)

    # init graph parameters
    groups = np.zeros((m, n), dtype=np.int32, order='F') + 1

    # define groups
    # idx=0 means no group on that pixel
    groupIdx = 2
    for j in range(0, n - b + 1, b):
        for i in range(0, m - a + 1, a):
            groups[i:(i + a), j:(j + b)] = groupIdx
            groupIdx += 1

    return groups.flatten(order='F')


def get_proximal_graph_nonoverlap(img_shape, batch_shape):
    if len(img_shape) != 2 or len(batch_shape) != 2:
        raise "Input lengths are incorrect"

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

    indptr = [0] * (numGroup+1)  # number of elements in each col
    indices = []

    # define groups
    groupIdx = 0
    for j in range(0, n - b + 1, b):
        for i in range(0, m - a + 1, a):
            varsIdx = get_vars_idx_top_left(i, j, (a,b), img_shape)
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

    indptr = [0] * (numGroup+1) # number of elements in each col
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
        indptr[groupIdx+1] = indptr[groupIdx] + len(varsIdx)
        indices.extend(varsIdx)

    data = np.full(len(indices), True)
    groups_var = ssp.csc_matrix((data, indices, indptr), shape=(m * n, numGroup), dtype=bool)

    graph = {'eta_g': eta_g, 'groups': groups, 'groups_var': groups_var}

    return graph


def inexact_alm_rpca(D0, delta=1, use_sv_prediction=False):
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
            print('CONVERGED')
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
    background_lambda = 5e2 * lambda_param

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
        S = prox_by_frame(G_S, lambda_param / mu, graphs)
        apply_background_shrinkage_operator(G_S, S, background_lambda / mu, background_masks)

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


def apply_morph_ops(input, footprint_name='disk'):
    im_height = input.shape[0]

    footprint_dilation = get_footprint(footprint_name, 0.05 * im_height)
    footprint_closing = get_footprint(footprint_name, 0.05 * im_height)
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


def build_improved_LSD_graphs(D, original_shape, weights, delta=1.0):
    print(f'delta = {delta:f}')

    L, S = inexact_alm_rpca(D, delta=delta)[:2]  # rpca
    # L[1], S[1] = inexact_alm_lsd_graph(D, graph_nonoverlap, delta)[:2]  # graphs
    # L[1], S[1] = inexact_alm_lsd_flat(D, groups_nonoverlap, delta)[:2]  # groups

    # mask S and reshape back to 3d array
    S_mask = foreground_mask(D, L, S, sigmas_from_mean=2) \
        .reshape(original_shape, order='F')

    # apply morphological operations
    S_mask_morph = apply_morph_ops(S_mask)
    weight_mask = merge_masks((S_mask, S_mask_morph), weights)

    mask_percent = calc_mask_percent(weight_mask) * 100
    print(f'mask percentage: {mask_percent:.2f}%')

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
        graphs = Parallel(n_jobs=2)(delayed(get_proximal_graph_group_centers) \
                                        (weight_mask[:, :, i].shape, group_radius, weight_mask[:, :, i]) for i in
                                    range(weight_mask.shape[-1]))
    else:
        graphs = [get_proximal_graph_group_centers(weight_mask[:, :, i].shape,
                                                   group_radius,
                                                   group_centers=weight_mask[:, :, i])
                  for i in range(weight_mask.shape[-1])]
    t1 = time.time()
    print(f'Graphs time: {t1-t0:.2f}s')

    background_masks = [(weight_mask[:, :, i] < 0).flatten(order='F') for i in range(weight_mask.shape[-1])]

    return graphs, background_masks


def LSD_improved(ImData0, frame_start=0, frame_end=47, downsample_ratio=1, delta=1):
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

    # reshape so that each frame is a column
    D = ImData2.reshape((np.prod(frame_size), frames), order='F')

    # build graphs
    weights = (1, 1.5)
    graphs, background_masks = build_improved_LSD_graphs(D, original_downsampled_shape, weights, delta=1.0)

    print('Running LSD...')
    L, S, iterations, converged = inexact_alm_lsd_with_background(D, graphs, background_masks)

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

    downsample_ratio = 4
    frame_start = 0
    frame_end = 47

    print('IMPROVED')
    t0 = time.time()
    S_mask_imp, L_recon_imp, ImData1_imp = LSD_improved(
        ImData0, frame_start=frame_start, frame_end=frame_end, downsample_ratio=downsample_ratio)
    t1 = time.time()

    print('ORIGINAL')
    t2 = time.time()
    S_mask, L_recon, ImData1 = LSD(
        ImData0, frame_start=frame_start, frame_end=frame_end, downsample_ratio=downsample_ratio)
    t3 = time.time()

    print(f'Run times: original: {t3 - t2:.2f}s, improved: {t1 - t0:.2f}s')

    N = 6
    video_length = ImData1.shape[2]
    subplots_samples([ImData1, S_mask, S_mask_imp, L_recon, L_recon_imp], range(0, video_length, video_length // N),
                     size_factor=4)


if __name__ == '__main__':
    print('START')

    enabled_str = '*ENABLED* :)' if USE_PARALLEL else 'DISABLED :('
    print(f'CORES: {num_cores}. Parallel processing {enabled_str}\n')

    start = time.time()
    main()
    end = time.time()
    print('DONE')
    print(f'ELAPSED TIME: {(end - start):.3f} seconds')