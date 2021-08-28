import cv2
import numpy as np
import glob
import os
import pandas as pd
from imageio import imread
import matplotlib.pylab as plt
from RobustPCA.rpca import RobustPCA  # install RobustPCA according to Readme.md
from RobustPCA.spcp import StablePCP

video_length = 100  # how many frames to process
downscale_factor = 1


def bitmap_to_mat(bitmap_seq):
    """
    bitmap_to_mat takes a list of image filenames and returns
    a numpy 4D array of those images, dtype is uint8
    matrix structure is (image_num,h,w,c)
    assumption: all images have the same dimensions
    """
    image_count = len(bitmap_seq)
    shape = None
    count = 0
    for bitmap_file in bitmap_seq:
        img = cv2.imread(bitmap_file)
        if shape is None:  # first image read
            shape = img.shape
            matrix = np.zeros((image_count, shape[0], shape[1], shape[2]), dtype=np.uint8)
        assert img.shape == shape
        matrix[count, :, :, :] = img
        count = count + 1
    return matrix


def execute():
    video_data = bitmap_to_mat(glob.glob("input/*.jpg")[:video_length])
    shape = video_data.shape

    # downscale video for faster decomposition, convert to float64
    px = (shape[1] // downscale_factor)
    py = (shape[2] // downscale_factor)
    video_data = video_data[:, ::downscale_factor, ::downscale_factor, :].reshape(video_length, px*py, 3)
    video_data = video_data.astype(np.float64)

    # build RPCA object and allocate space for low rank and sparse matrices
    # ahead of time
    rpca = RobustPCA(max_iter=500, use_fbpca=True, max_rank=1, verbose=True)
    L_video = np.zeros(video_data.shape, dtype=video_data.dtype)
    S_video = np.zeros(video_data.shape, dtype=video_data.dtype)

    # perform RPCA for each channel separately
    for i in range(3):
        rpca.fit(video_data[:, :, i])
        L_video[:, :, i] = rpca.get_low_rank()
        S_video[:, :, i] = rpca.get_sparse()

    open('L_video_bin_dump.bin', 'wb').write(L_video.tobytes())
    open('S_video_bin_dump.bin', 'wb').write(S_video.tobytes())


if __name__ == '__main__':
    print("Running in: "+str(os.getcwd()))
    execute()
