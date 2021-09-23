import argparse

import cv2
import numpy as np
import glob
import os
from datetime import datetime
import time
from RobustPCA.rpca import RobustPCA
from utils import *
import shutil

video_length = 200  # how many frames to process
downscale_factor = 1


def compute_RPCA(image_array, grayscale, max_error):
    """
    given a video array, compute a decomposition of each image frame and each color
    into a low rank matrix and a sparse matrix.
    The function also saves the input and output matrices to files and stores a log
    of computation
    """
    shape = image_array.shape
    length = shape[0]

    # allocate storage
    L_array = np.zeros(image_array.shape, dtype=image_array.dtype)
    S_array = np.zeros(image_array.shape, dtype=image_array.dtype)

    # prepare RPCA object.
    rpca = RobustPCA(max_iter=200000, use_fbpca=True, max_rank=1, tol=max_error, verbose=False)

    # start log
    # perform RPCA for each image and each color channel separately
    if grayscale:
        for i in range(length):
            # compute decomposition and store
            print(f"Processing image {i} out of {length}")
            rpca.fit(image_array[i, :, :])
            L_array[i, :, :] = rpca.get_low_rank()
            S_array[i, :, :] = rpca.get_sparse()

    else:
        for i in range(length):
            for c in range(3):
                print(f"Processing image {i} out of {length}, Performing {c} th fit of 3 ")
                # compute decomposition and store
                rpca.fit(image_array[i, :, :, c])
                L_array[i, :, :, c] = rpca.get_low_rank()
                S_array[i, :, :, c] = rpca.get_sparse()

                # print log for this iteration

    return L_array, S_array


def executeSaliencyRPCA(ImData, downsample_ratio, grayscale_workmode=True, grayscale_input=True):
    video_data = ImData
    if grayscale_workmode and not grayscale_input:
        video_grayscale_data = np.zeros(video_data.shape[0:3], dtype=video_data.dtype)
        for i in range(video_data.shape[0]):
            video_grayscale_data[i, :, :] = cv2.cvtColor(video_data[i, :, :], cv2.COLOR_BGR2GRAY)
        video_data = video_grayscale_data
    # compute X-T and Y-T frames
    xt_plane = np.copy(video_data)
    yt_plane = np.copy(video_data)

    if grayscale_workmode:
        xt_plane = xt_plane.transpose([2, 1, 0])  # new order of axis relative to [t,h,w]
        yt_plane = yt_plane.transpose([1, 2, 0])
    else:
        xt_plane = xt_plane.transpose([2, 1, 0, 3])  # new order of axis relative to [t,h,w,c]
        yt_plane = yt_plane.transpose([1, 2, 0, 3])

    if grayscale_workmode:
        xt_plane = resize_with_cv2(xt_plane, 1 / downsample_ratio).astype(np.float64)
        yt_plane = resize_with_cv2(yt_plane, 1 / downsample_ratio).astype(np.float64)
    else:
        xt_plane = xt_plane[:, ::downscale_factor, ::downscale_factor, :].astype(np.float64)
        yt_plane = yt_plane[:, ::downscale_factor, ::downscale_factor, :].astype(np.float64)

    # save_images(xt_plane, args.output+f"output_xt_{idx}", grayscale_workmode)
    # save_images(yt_plane, args.output+f"output_yt_{idx}", grayscale_workmode)

    # compute decomposition for X-T plane
    print("Starting xt RPCA")

    xt_lowrank, xt_sparse = compute_RPCA(xt_plane, grayscale_workmode,
                                         xt_plane.shape[1] * xt_plane.shape[2] * 0.0001)
    # save_images(xt_sparse, args.output+f"output_xt_sparse_{idx}", grayscale_workmode)
    # save_images(xt_lowrank, args.output+f"output_xt_lowrank_{idx}", grayscale_workmode)

    # compute decomposition for Y-T plane
    print("Starting yt RPCA")
    yt_lowrank, yt_sparse = compute_RPCA(yt_plane, grayscale_workmode,
                                         yt_plane.shape[1] * yt_plane.shape[2] * 0.0001)
    # save_images(yt_sparse, args.output+f"output_yt_sparse_{idx}", grayscale_workmode)
    # save_images(yt_lowrank, args.output+f"output_yt_lowrank_{idx}", grayscale_workmode)
    return xt_lowrank, xt_sparse, yt_lowrank, yt_sparse


def main(args, idx):
    prepare_saliency_folders(args.output, idx)
    np.random.seed(0)

    # set print precision to 2 decimal points
    np.set_printoptions(precision=2)

    # import video
    # using dtype=np.float64 to allow normalizing. use np.uint8 if not needed.
    _, video_data = import_video_as_frames(args.input, args.frame_start, args.frame_end)
    original_shape = video_data

    xt_lowrank, xt_sparse, yt_lowrank, yt_sparse = executeSaliencyRPCA(video_data,
                                                                       downsample_ratio=args.downscale,
                                                                       grayscale_workmode=True,
                                                                       grayscale_input=True)

    np.save(args.output + "xt_lowrank", xt_lowrank)
    np.save(args.output + "xt_sparse", xt_sparse)
    np.save(args.output + "yt_lowrank", yt_lowrank)
    np.save(args.output + "yt_sparse", yt_sparse)


if __name__ == '__main__':
    idx = 0
    parser = argparse.ArgumentParser(description='run Saliency RPCA')
    parser.add_argument('--input', type=str, default=".", help='path to input folder with jpg frames')
    parser.add_argument('--output', type=str, default=".", help='path to output folder to store binary results')
    parser.add_argument('--frame_start', type=int, default=0, help='start frame index')
    parser.add_argument('--frame_end', type=int, default=2000, help='end frame index, inclusive')
    parser.add_argument('--downscale', type=int, default=1, help='downscale factor')
    parser.add_argument('--plot', type=bool, default=False, help='plot or not')
    args = parser.parse_args()

    print('START')
    write_log_to_file(args.output + 'computelog.txt', args)
    start = time.time()
    main(args, idx)
    end = time.time()
    print('DONE')
    print(f'ELAPSED TIME: {(end - start):.3f} seconds')
    with open(args.output + 'computelog.txt', 'a') as f:
        f.write(f'ELAPSED TIME: {(end - start):.3f} seconds')
