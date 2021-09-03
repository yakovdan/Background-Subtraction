import cv2
import numpy as np
from scipy.ndimage.filters import convolve


def gkern(l=10, sig=1.):
    """
    creates a 3d gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy) + np.square(zz)) / np.square(sig))

    return kernel / np.sum(kernel)


def load_sparse_array(filename, dimensions):
    """
    load a numpy array from a binary file (filename)
    and arrange it into an array with the provided dimensions
    """
    f = open(filename, 'rb')
    raw_array = f.read()
    f.close()
    np_array = np.frombuffer(raw_array, dtype=np.float64).reshape(dimensions)
    return np_array


def build_sparse_xt_cube(sparse_xt):
    """
    takes a sparse_xt array and converts it back into video-array order.
    also applies abs and converts to contiguous c array for performance
    """
    cube = np.abs(sparse_xt.transpose([2, 1, 0, 3]))
    return np.ascontiguousarray(cube)


def build_sparse_yt_cube(sparse_yt):
    """
    takes a sparse_xt array and converts it back into video-array order.
    also applies abs and converts to contiguous c array for performance
    """
    cube = np.abs(sparse_yt.transpose([2, 0, 1, 3]))
    return np.ascontiguousarray(cube)


def build_final_cube(sparse_xt_cube, sparse_yt_cube):
    """
    takes a video built from sparse_xt and another one from sparse_yt
    and combines them into a final Scube by element wise multiplication
    and normalization
    """
    assert sparse_xt_cube.shape == sparse_yt_cube.shape
    assert sparse_xt_cube.dtype == np.float64
    assert sparse_yt_cube.dtype == np.float64
    cube = np.multiply(sparse_xt_cube, sparse_yt_cube) # elementwise
    sum_of_cube = np.sum(cube)
    return np.divide(cube, sum_of_cube)


def adaptive_threshold(cube):
    """
    given a video cube, compute mean and std deviation per channel
    set the value to 1 if more than one std deviation above mean
    """
    binary_cube = np.zeros(cube.shape, dtype=np.uint8)
    for c in range(3):
        mean = np.mean(cube[:, :, :, c])
        std = np.std(cube[:, :, :, c])
        idx = (cube[:, :, :, c] > mean+std)
        binary_cube[idx, c] = 1

    return binary_cube


def output_video(video_array, path):
    """
    takes a video numpy array and a path, formats video for output and writes it frame by frame
    """
    video_out_array = video_array * 255
    video_out_array = video_out_array.astype(np.uint8)
    for i in range(video_out_array.shape[0]):
        cv2.imwrite(path+f"/output_sparse_frame_{i}.bmp", cv2.cvtColor(video_out_array[i, :, :, :], cv2.COLOR_BGR2RGB))


def execute():
    sparse_xt = load_sparse_array('S_video_bin_dump_xt_plane.bin', (320, 240, 100, 3))
    sparse_yt = load_sparse_array('S_video_bin_dump_yt_plane.bin', (240, 320, 100, 3))
    sparse_xt_cube = build_sparse_xt_cube(sparse_xt)
    sparse_yt_cube = build_sparse_yt_cube(sparse_yt)
    sparse_cube = build_final_cube(sparse_xt_cube, sparse_yt_cube)
    smooth_sparse_cube = np.zeros(sparse_cube.shape, dtype=sparse_cube.dtype)

    for c in range(3):
        smooth_sparse_cube[:, :, :, c] = convolve(sparse_cube[:, :, :, c], gkern(), mode='reflect')
    binary_cube = adaptive_threshold(smooth_sparse_cube)
    output_video(binary_cube, "binary_video")
    print(np.sum(smooth_sparse_cube))


if __name__ == "__main__":
    execute()