import argparse
import sys

import numpy as np
from numpy import linalg as LA

from inexact_alm_lsd import subplots_samples
from motion_saliency_check import run_motion_saliency_check
from utils import *
import time


def block_shrinkage_operator(G, blocks_by_frame, lambdas_by_frame, mu, non_block_lambda):
    """
    G[pixel, frame] - input matrix
    blocks_by_frame[frame, group, mask_pixel] - binary column masks
    lambdas_by_frame[frame, group] - lambda for each block
    mu - number
    """
    result = np.zeros_like(G)
    for frame_idx in range(len(blocks_by_frame)):  # run over frames
        blocks = blocks_by_frame[frame_idx]  # get the blocks of that frame
        lambdas = lambdas_by_frame[frame_idx]

        non_block = np.full(G.shape[0], True)  # a negative mask of all blocks
        for block_idx in range(len(blocks)):  # run over blocks
            # assuming n overlap between groups
            block = blocks[block_idx]
            epsilon = lambdas[block_idx] / mu

            # mark False where blocks are found
            non_block[block] = False

            block_in_G = G[block, frame_idx]
            result[block, frame_idx] = np.maximum(1 - epsilon / LA.norm(block_in_G, ord=2), 0) * block_in_G

        # use a big lambda for the area around the blocks
        non_block_in_G = G[non_block, frame_idx]
        non_block_epsilon = non_block_lambda / mu
        result[non_block, frame_idx] = np.maximum(1 - non_block_epsilon / LA.norm(non_block_in_G, ord=2), 0) * non_block_in_G

    return result


def inexact_alm_group_sparse_RPCA(D0, blocks_by_frame, lambdas_by_frame, delta=10, use_sv_prediction=True):
    # make sure D is in fortran order
    if not np.isfortran(D0):
        print('D_in is not in Fortran order')
        D = np.asfortranarray(D0)
    else:
        D = D0

    m, n = D.shape
    d = np.min(D.shape)

    lambda_param = (np.sqrt(np.max((m, n))) * delta) ** (-1)
    non_block_lambda = 8e2 * lambda_param
    MAX_RANK = d + 1

    # initialize
    Y = D
    norm_two = LA.norm(Y, ord=2)
    norm_inf = LA.norm(Y, ord=np.inf) / lambda_param
    dual_norm = np.max((norm_two, norm_inf))
    Y = Y / dual_norm

    mu = 1.25 / norm_two  # can be tuned
    rho = 1.6
    tol_out = 1e-7

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

        if svp == 0:
            print("L reached rank 0")
            break

        # predicting the number of s.v bigger than 1/mu
        if use_sv_prediction:
            sv = svp + 1 if svp < sv else min(svp + round(0.05 * d), d)

        L = svd_reconstruct(u[:, :svp], s[:svp] - 1 / mu, vh[:svp, :], order='F')

        # SOLVE FOR S
        G_S = D - L + Y / mu  # Algorithm line 7
        S = block_shrinkage_operator(G_S, blocks_by_frame, lambdas_by_frame, mu, non_block_lambda)

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
            print('DIDNT CONVERGE')
            break
        elif svp >= MAX_RANK:
            print('L reached rank ' + str(MAX_RANK))
            break

    return L, S, iter_out, converged


def check_BS_operator():
    frame_size, frames = (5, 3)
    G = np.array(range(frame_size * frames)).reshape((frame_size, frames), order='F')
    np.random.shuffle(G)
    print(G)

    blocks_by_frame = [
        [[0, 1], [3, 4]],
        [[0]],
        [[1], [2, 3], [4]]
    ]
    print(blocks_by_frame)

    lambdas_by_frame = [
        [1, 2],
        [3],
        [4, 5, 6]
    ]
    print(lambdas_by_frame)

    mu = 1000

    block_shrinkage_result = block_shrinkage_operator(G, blocks_by_frame, lambdas_by_frame, mu, large_val=100)
    print(block_shrinkage_result)


def load_data(video_length, cut_length=None, path_to_data=None, path_to_input=None):
    image_mean = 0.4233611323018794

    if cut_length is None:
        cut_length = video_length

    if path_to_data is not None:
        data_root = path_to_data
    else:
        data_root = "."

    lowrank_mat = load_mat_from_bin(data_root+'/highway_200frames/Lowrank1_highway.bin',
                                    np.float64, (320, 240, video_length))[:, :, :cut_length].transpose((1, 0, 2))
    sparse_mat = load_mat_from_bin(data_root+'/highway_200frames/Sparse1_highway.bin',
                                   np.float64, (320, 240, video_length))[:, :, :cut_length].transpose((1, 0, 2))
    lowrank_reconstructed = lowrank_mat + image_mean

    # sparse_cube is in [t,h,w] order so transpose
    sparse_cube = load_mat_from_bin(data_root+'./highway_200frames/sparse_cube_200.bin',
                                    np.float64, (320, 240, video_length))[:, :, :cut_length].transpose((1, 0, 2))
    if path_to_input is not None:
        input_root = path_to_input
    else:
        input_root = "."
    video_list = glob.glob(input_root+"/input/*.jpg")
    video_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    video_list = video_list[:cut_length]

    ##############################
    # Load frames and preprocess #
    ##############################
    Data = bitmap_to_mat(video_list, True).transpose((1, 2, 0)).astype(np.float64)  # t, h, w -> h, w, t
    normalizeImage(Data)  # [0, 255] -> [0,1]
    DataMean = np.mean(Data)
    Data -= DataMean
    shape = Data.shape

    return Data, DataMean, lowrank_mat, sparse_mat, sparse_cube


def main(path_to_data=None, path_to_input=None, path_to_output=None):

    video_length = 200
    cut_length = 200
    np.random.seed(0)

    # set print precision to 2 decimal points
    np.set_printoptions(precision=2)

    delta = 10

    print("LOAD DATA")
    Data, DataMean, lowrank_mat, sparse_mat, sparse_cube = load_data(video_length, cut_length, path_to_data, path_to_input)

    groups_by_frame, weights_by_frame = run_motion_saliency_check(Data, lowrank_mat, sparse_mat, sparse_cube, delta=delta)

    original_shape = Data.shape
    D = Data.reshape((320 * 240, cut_length), order='F')

    print("GROUP SPARSE RPCA")
    L, S, iterations, converged = inexact_alm_group_sparse_RPCA(D, groups_by_frame, weights_by_frame, delta=delta)

    # mask S and reshape back to 3d array
    S_mask_2 = foreground_mask(D, L, S, sigmas_from_mean=2).reshape(original_shape, order='F')
    S_mask_3 = foreground_mask(D, L, S, sigmas_from_mean=3).reshape(original_shape, order='F')
    L_recon = L.reshape(original_shape, order='F') + DataMean
    Data += DataMean

    S_reshaped = S.reshape(original_shape, order='F')
    normalizeImage(S_reshaped)

    print('Plotting...')
    N = 6
    subplots_samples((S_mask_3, S_mask_2, S_reshaped, L_recon, Data), range(0, cut_length, cut_length // N), size_factor=2)

    if path_to_output is not None:
        output_root = path_to_output
    else:
        output_root = "."
    output_result_bitmap_seq(output_root+'/output/', Data, L_recon, S_reshaped, S_mask_2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run group sparse')
    parser.add_argument('--input', type=str, default=".", help='Path to input folder with jpg frames')
    parser.add_argument('--output', type=str, default=".", help='Path to output folder where output frames will be saved')
    parser.add_argument('--data', type=str, default=".", help='Path to data folder where binary matrices are stored')
    args = parser.parse_args()
    print('START')
    start = time.time()
    main(args.data, args.input, args.output)
    end = time.time()
    print('DONE')
    print(f'ELAPSED TIME: {(end-start):.3f} seconds')
