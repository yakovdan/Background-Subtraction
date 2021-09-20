from collections import OrderedDict

import numpy as np
import cv2
import os
import sys
from utils import *
from inexact_alm_lsd import *
from computeRPCADecomposition import *
from motion_saliency_check import *
from computeSCube import *


def main(run_index, video_length):
    all_groups = []
    groups_by_frame = []
    weights_by_frame = []
    image_mean = 0.4233611323018794
    np.random.seed(0)

    # set print precision to 2 decimal points
    np.set_printoptions(precision=2)

    # set video start frame, end frame and downsample ratio
    frame_start = 0
    frame_end = video_length
    downsample_ratio = 1

    # read video frames
    #ImData0, c_format_data = import_video_as_frames()

    # prepare saliency output folders
    #prepare_saliency_folders(run_index)

    # run LSD
    # lowrank_result, sparse_result, lowrank_reconstructed, sparse_mask, data_mean = LSD(ImData0, frame_start=frame_start,
    #                                                                                    frame_end=frame_end,
    #                                                                                    downsample_ratio=downsample_ratio)
    # # compute saliency RPCA in xt and yt planes.
    # # requires video matrix in [t, h, w] format, hence the transpose
    # xt_lowrank, xt_sparse, yt_lowrank, yt_sparse = executeSaliencyRPCA(c_format_data, run_index, downsample_ratio=downsample_ratio)

    lowrank_mat = load_mat_from_bin('./highway_200frames/Lowrank1_highway.bin', np.float64, (320, 240, video_length))
    sparse_mat = load_mat_from_bin('./highway_200frames/Sparse1_highway.bin', np.float64, (320, 240, video_length))
    lowrank_reconstructed = lowrank_mat + image_mean


    #sparse_cube = computeSCube('./highway_200frames/S_video_bin_dump_xt_plane.bin',
    #                           './highway_200frames/S_video_bin_dump_yt_plane.bin')

    #sparse_cube is in [t,h,w] order so transpose
    #sparse_cube = np.ascontiguousarray(sparse_cube.transpose((2, 1, 0))) # now it's [320,240,200], float64
    #save_mat_to_bin(sparse_cube, 'sparse_cube_200.bin')
    sparse_cube = load_mat_from_bin('./highway_200frames/sparse_cube_200.bin', np.float64, (320, 240, video_length))
    video_list = glob.glob("./input/*.jpg")
    video_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    video_list = video_list[:video_length]

    ##############################
    # Load frames and preprocess #
    ##############################
    Data = bitmap_to_mat(video_list, True).transpose((2, 1, 0)).astype(np.float64) # t, h, w -> w, h, t
    Data -= np.min(Data)
    Data *= (1.0 / np.max(Data))  # [0, 255] -> [0,1]
    DataMean = np.mean(Data)
    Data -= DataMean
    shape = Data.shape

    groups_by_frame, weights_by_frame = run_motion_saliency_check(Data, lowrank_mat, sparse_mat, sparse_cube)

if __name__ == "__main__":
    run_index = 0
    print("Starting!")
    main(run_index, 200)
    print("Done!")
