from collections import OrderedDict

import numpy as np
import cv2
import os
import sys
from utils import *
from inexact_alm_lsd import *
from computeRPCADecomposition import *
import hashlib
from group_sparse_RPCA import *
from motion_saliency_check import *
from computeSCube import *


def main(lsd_path, saliency_path, output_path, frame_count, frame_start=0):

    cut_length = frame_count
    np.random.seed(0)

    # set print precision to 2 decimal points
    np.set_printoptions(precision=2)
    delta = 10

    # set video start frame, end frame and downsample ratio
    print("start loading data")
    frame_end = frame_count-1
    sparse_binary_mat = np.load(f"{lsd_path}/sparse.bin.npy")
    fullscale_video = np.load(f"{lsd_path}/video_data.npy").astype(np.float64)
    xt_sparse = np.load(f"{saliency_path}/xt_sparse.npy")
    yt_sparse = np.load(f"{saliency_path}/yt_sparse.npy")

    print("done loading data")
    if fullscale_video.shape[0] * fullscale_video.shape[1] != sparse_binary_mat.shape[0] * sparse_binary_mat.shape[1]: # upscale required
        sparse_binary_mat_upscale = np.zeros(fullscale_video.shape, dtype=np.bool)

        for i in range(sparse_binary_mat.shape[2]):
            scaler_mat = np.ones((2, 2), dtype=np.bool)
            sparse_binary_mat_upscale[:, :, i] = np.kron(sparse_binary_mat[:, :, i], scaler_mat)

        sparse_binary_mat = sparse_binary_mat_upscale

    assert sparse_binary_mat.shape == fullscale_video.shape

    hash_obj = hashlib.md5(saliency_path.encode())
    if os.path.isfile("sparse_cube"+hash_obj.hexdigest()+".npy"):
        sparse_cube = np.load("sparse_cube"+hash_obj.hexdigest()+".npy")
    else:
        sparse_cube = computeSCube(xt_sparse, yt_sparse)
        sparse_cube = np.ascontiguousarray(sparse_cube.transpose((1, 2, 0)))
        np.save("sparse_cube"+hash_obj.hexdigest(), sparse_cube)

    ##############################
    # Load frames and preprocess #
    ##############################
    video_mean = np.mean(fullscale_video)
    video_data_without_mean = fullscale_video - video_mean

    print("running motion check")
    groups_by_frame, weights_by_frame = run_motion_saliency_check(video_data_without_mean,
                                                                  sparse_binary_mat,
                                                                  sparse_cube)

    total_groups = 0
    for fr in groups_by_frame:
        total_groups += len(fr)

    print(f"total groups: {total_groups}")

    original_shape = video_data_without_mean.shape
    D = np.asfortranarray(video_data_without_mean.reshape(
                          video_data_without_mean.shape[0]*video_data_without_mean.shape[1],
                          video_data_without_mean.shape[2]),
                          order='F')

    print("GROUP SPARSE RPCA")
    L, S, iterations, converged = inexact_alm_group_sparse_RPCA(D, groups_by_frame, weights_by_frame, delta=delta)
    print("saving")
    # mask S and reshape back to 3d array
    S_mask_2 = foreground_mask(D, L, S, sigmas_from_mean=2)
    S_mask_2 = S_mask_2.reshape(original_shape, order='F')
    S_mask_3 = foreground_mask(D, L, S, sigmas_from_mean=3).reshape(original_shape, order='F')
    L_recon = L.reshape(original_shape, order='F') + video_mean


    S_reshaped = S.reshape(original_shape, order='F')
    normalizeImage(S_reshaped)

    # remove objects that are too small
    S_mask_2 = filter_sparse_map(S_mask_2)
    S_mask_3 = filter_sparse_map(S_mask_3)
    # print('Plotting...')
    # N = 6
    # subplots_samples((S_mask_3, S_mask_2, S_reshaped, L_recon, video_data), range(0, cut_length, cut_length // N), size_factor=2)

    output_result_bitmap_seq(output_path+'/final/', fullscale_video, L_recon, S_reshaped, S_mask_2)
    np.save(output_path+'/binarymask/S_mask2', S_mask_2)
    np.save(output_path+'/binarymask/S_mask3', S_mask_3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run Saliency RPCA')
    parser.add_argument('--lsd_path', type=str, default=".", help='path to lsd')
    parser.add_argument('--saliency_path', type=str, default=".", help='path to saliency')
    parser.add_argument('--output_path', type=str, default=".", help='path to output')
    parser.add_argument('--frame_count', type=int, default=0, help='frame_count')
    args = parser.parse_args()
    print("Starting!")
    main(args.lsd_path, args.saliency_path, args.output_path, args.frame_count)
    print("Done!")
