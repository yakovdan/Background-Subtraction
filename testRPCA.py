import cv2
import numpy as np
import glob
import os
import sys
# import pandas as pd
# from imageio import imread
# import matplotlib.pylab as plt
from RobustPCA.rpca import RobustPCA  # install RobustPCA according to Readme.md
# from RobustPCA.spcp import StablePCP

video_length = 100  # how many frames to process
downscale_factor = 1


def save_images(image_array, path):
    num_images = image_array.shape[0]
    filenames = [path+f"/output_image{i}.bmp" for i in range(num_images)]
    for i in range(num_images):
        cv2.imwrite(filenames[i], image_array[i, :, :, :])


def prepare_image_array_for_rpca(image_array, resize_factor):
    """
    downscale video for faster decomposition, convert to float64 and column stack
    """
    shape = image_array.shape
    vid_length = shape[0]
    px = (shape[1] // resize_factor)
    py = (shape[2] // resize_factor)
    image_array = image_array[:, ::resize_factor, ::resize_factor, :].reshape(vid_length, px*py, 3)
    image_array = image_array.astype(np.float64)
    return image_array

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


def compute_RPCA(image_array, name):
    shape = image_array.shape
    size = shape[0] * shape[1] * shape[2]
    #rpca = RobustPCA(max_iter=500, lamb=1/size**(1/3), use_fbpca=True, max_rank=1, verbose=True)
    #rpca = RobustPCA(max_iter=1000, lamb=(1/(size**(1/3))), use_fbpca=True,  verbose=True)
    #rpca = RobustPCA(max_iter=500, lamb=(1/(size**(1/3))), use_fbpca=False,  verbose=True)
    rpca = RobustPCA(max_iter=500, lamb=0.3, use_fbpca=False,  verbose=True)
    L_array = np.zeros(image_array.shape, dtype=image_array.dtype)
    S_array = np.zeros(image_array.shape, dtype=image_array.dtype)

    # perform RPCA for each channel separately
    for i in range(3):
        print(f"Performing {i}th fit of 3")
        rpca.fit(image_array[:, :, i])
        L_array[:, :, i] = rpca.get_low_rank()
        S_array[:, :, i] = rpca.get_sparse()

    open("L_video_bin_dump_"+name+".bin", 'wb').write(L_array.tobytes())
    open("S_video_bin_dump_"+name+".bin", 'wb').write(S_array.tobytes())
    open("image_array_dump"+name+".bin", "wb").write(image_array.tobytes())
    return L_array, S_array

def execute():
    video_data = bitmap_to_mat(glob.glob("./input/*.jpg")[:video_length:1])

    xt_plane = np.copy(video_data)
    xt_plane = xt_plane.transpose([2, 1, 0, 3])  # new order of axis relative to [t,h,w,c]
    yt_plane = np.copy(video_data)
    yt_plane = yt_plane.transpose([1, 2, 0, 3])
    save_images(xt_plane, "output_xt")
    save_images(xt_plane, "output_yt")

    xt_plane = prepare_image_array_for_rpca(xt_plane, downscale_factor)
    yt_plane = prepare_image_array_for_rpca(yt_plane, downscale_factor)

    print("Starting xt RPCA")
    xt_lowrank, xt_sparse = compute_RPCA(xt_plane, "xt_plane")
    print("Starting yt RPCA")
    yt_lowrank, yt_sparse = compute_RPCA(yt_plane, "yt_plane")




if __name__ == '__main__':
    print("Running in: "+str(os.getcwd()))
    execute()
