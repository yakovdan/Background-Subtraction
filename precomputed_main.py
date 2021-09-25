from collections import OrderedDict

import numpy as np
import cv2
import os
import sys
from utils import *
from inexact_alm_lsd import *
from computeRPCADecomposition import *
from group_sparse_RPCA import *
from motion_saliency_check import *
from computeSCube import *


def main(root_path, scale_path, path_to_output,  frame_count, frame_start=0):
    cut_length = frame_count
    np.random.seed(0)
    # set print precision to 2 decimal points
    np.set_printoptions(precision=2)
    delta = 10
    # set video start frame, end frame and downsample ratio

    frame_end = frame_count+frame_count-1
    downsample_ratio = 4 if scale_path == 'smallscale/' else 1
    video_mean, _ = parse_numerical_values(root_path+"LSD/"+scale_path+"numerical_values.txt")

    lowrank_mat = np.load(root_path+"LSD/"+scale_path+"lowrank.npy")
    sparse_mat = np.load(root_path+"LSD/"+scale_path+"sparse.npy")
    lowrank_reconstructed = lowrank_mat + video_mean

    xt_sparse = np.load(root_path+"Saliency/"+scale_path+"xt_sparse.npy")
    xt_lowrank = np.load(root_path+"Saliency/"+scale_path+"xt_lowrank.npy")
    yt_sparse = np.load(root_path+"Saliency/"+scale_path+"yt_sparse.npy")
    yt_lowrank = np.load(root_path+"Saliency/"+scale_path+"yt_lowrank.npy")

    #sparse_cube = computeSCube_frompaths('./highway_200frames/S_video_bin_dump_xt_plane.bin',
    #                           './highway_200frames/S_video_bin_dump_yt_plane.bin')

    sparse_cube = computeSCube(xt_sparse, yt_sparse)


    #sparse_cube is in [t,h,w] order so transpose
    #sparse_cube = np.ascontiguousarray(sparse_cube.transpose((2, 1, 0))) # now it's [320,240,200], float64
    #save_mat_to_bin(sparse_cube, 'sparse_cube_200.bin')
    #sparse_cube = load_mat_from_bin('./highway_200frames/sparse_cube_200.bin', np.float64, (320, 240, video_length))
    #video_list = glob.glob("./input/*.jpg")
    #video_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    #video_list = video_list[:video_length]

    ##############################
    # Load frames and preprocess #
    ##############################
    video_data = np.load(root_path+"LSD/"+scale_path+"data.npy")
    ########### subtracting mean here because this was forgotten in the LSD source
    video_data = video_data - np.mean(video_data)
    sparse_cube = np.ascontiguousarray(sparse_cube.transpose((1, 2, 0)))
    groups_by_frame, weights_by_frame = run_motion_saliency_check(video_data, lowrank_mat, sparse_mat, sparse_cube)
    original_shape = video_data.shape
    D = np.asfortranarray(video_data.reshape(video_data.shape[0]*video_data.shape[1], video_data.shape[2]))

    print("GROUP SPARSE RPCA")
    L, S, iterations, converged = inexact_alm_group_sparse_RPCA(D, groups_by_frame, weights_by_frame, delta=delta)

    # mask S and reshape back to 3d array
    S_mask_2 = foreground_mask(D, L, S, sigmas_from_mean=2)
    S_mask_2 = S_mask_2.reshape(original_shape, order='C')
    S_mask_3 = foreground_mask(D, L, S, sigmas_from_mean=3).reshape(original_shape, order='C')
    L_recon = L.reshape(original_shape, order='C') + video_mean
    video_data += video_mean

    S_reshaped = S.reshape(original_shape, order='C')
    normalizeImage(S_reshaped)

    # remove objects that are too small
    S_mask_2 = filter_sparse_map(S_mask_2)

    print('Plotting...')
    N = 6
    subplots_samples((S_mask_3, S_mask_2, S_reshaped, L_recon, video_data), range(0, cut_length, cut_length // N), size_factor=2)

    if path_to_output is not None:
        output_root = path_to_output
    else:
        output_root = "."
    output_result_bitmap_seq(output_root+'/output/', video_data, L_recon, S_reshaped, S_mask_2)
    np.save(output_root+'/output/S_mask2', S_mask_2)
if __name__ == "__main__":
    root_path = 'D:/Masters/ProcessedData/highway_1700/'
    scale_path = 'smallscale/'
    frame_count = 1700
    output_path = "./output_1700/"
    print("Starting!")
    main(root_path, scale_path, output_path,  frame_count)
    print("Done!")
