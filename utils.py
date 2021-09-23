import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob
from functools import reduce
from numpy import linalg as LA
from scipy.sparse.linalg import svds
from datetime import datetime


def create_folder(name):
    if os.path.exists(name):
        shutil.rmtree(name)

    os.mkdir(name)


def prepare_saliency_folders(output_path, idx):
    create_folder(output_path + f"output_xt_{idx}")
    create_folder(output_path + f"output_yt_{idx}")
    create_folder(output_path + f"output_xt_sparse_{idx}")
    create_folder(output_path + f"output_xt_lowrank_{idx}")
    create_folder(output_path + f"output_yt_sparse_{idx}")
    create_folder(output_path + f"output_yt_lowrank_{idx}")


def bitmap_to_mat(bitmap_seq, grayscale=True):
    """
    bitmap_to_mat takes a list of image filenames and returns
    a numpy 4D array of those images, dtype is uint8
    matrix structure is (h,w,t)
    assumption: all images have the same dimensions
    """
    image_count = len(bitmap_seq)
    shape = None
    count = 0
    for bitmap_file in bitmap_seq:
        img = cv2.imread(bitmap_file)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if shape is None:  # first image read
            shape = img.shape
            if grayscale:
                matrix = np.zeros((shape[0], shape[1], image_count), dtype=np.uint8)
            else:
                matrix = np.zeros((shape[0], shape[1], shape[2], image_count), dtype=np.uint8)
        assert img.shape == shape
        if grayscale:
            matrix[:, :, count] = img
        else:
            matrix[:, :, :, count] = img
        count = count + 1
    return matrix


def import_video_as_frames(path, start, end):
    """
    import video
    using dtype=np.float64 to allow normalizing. use np.uint8 if not needed.
    sort frames in ascending number order because glob can return filenames in any order

    video_data is in [t,h,w] order and is a c-order array
    ImData0 is in [h,w, t] order and is a fortran array

    """
    frames_list = glob.glob(path+'*.jpg')
    frames_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    frames_list = frames_list[start:end]
    bitmap_seq = bitmap_to_mat(frames_list, grayscale=True).astype(np.float64)
    video_data = np.ascontiguousarray(bitmap_seq.transpose((2, 0, 1)))
    ImData0 = np.asfortranarray(bitmap_seq)
    return ImData0, video_data


def save_images(image_array, path, grayscale):
    """
    given an video array and a folder path, save each image frame to file
    by iterating over the first axis of the array
    """

    num_images = image_array.shape[0]
    filenames = [path+f"/output_image{i}.bmp" for i in range(num_images)]
    for i in range(num_images):
        if grayscale:
            cv2.imwrite(filenames[i], image_array[i, :, :])
        else:
            cv2.imwrite(filenames[i], image_array[i, :, :, :])


def plot_errors(errors_list, filename, display=False, log_scale=True):
    iterations = np.arange(1, len(errors_list)+1)
    if log_scale:
        data = np.log(np.array(errors_list))
    else:
        data = np.array(errors_list)
    plt.plot(iterations, data)
    plt.xlabel('iterations')
    plt.ylabel('errors')
    plt.title('Decomposition error over iterations')
    plt.savefig(filename)
    if display:
        plt.show()


def resize_with_cv2_timefirst(images, ratio):
    result_size = [int(np.ceil(images.shape[i] * ratio)) for i in [1, 2]]
    T = images.shape[0]
    result = np.empty([T]+result_size)
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    for t in range(T):
        result[t, :, :] = cv2.resize(images[t, :, :], result_size[::-1], interpolation=interpolation)
    return result


def resize_with_cv2(images, ratio):
    result_size = [int(np.ceil(images.shape[i] * ratio)) for i in [0, 1]]
    T = images.shape[2]
    result = np.empty(result_size + [T])
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    for t in range(T):
        result[:, :, t] = cv2.resize(images[:, :, t], result_size[::-1], interpolation=interpolation)
    return result


def foreground_mask(D, L, S, sigmas_from_mean=2):
    S_abs = np.abs(S)
    m = np.max(S_abs)
    S_back_temp = S_abs < 0.5 * m
    S_diff = np.abs(D - L) * S_back_temp
    positive_S_diff = S_diff[S_diff > 0]
    mu_s = np.mean(positive_S_diff)
    sigma_s = np.std(positive_S_diff)
    th = mu_s + sigmas_from_mean * sigma_s
    mask = S_abs > th
    return mask


def load_mat_from_bin(filename, dtype, shape):
    """
    load a numpy array from a binary file (filename)
    and arrange it into an array with the provided dimensions and data type
    """

    f = open(filename, 'rb')
    byte_array = f.read()
    f.close()
    np_array = np.frombuffer(byte_array, dtype=dtype)
    np_array = np_array.reshape(shape)
    return np_array


def save_mat_to_bin(matrix, filename):
    """
    saves matrix to filename
    """
    f = open(filename, 'wb')
    f.write(matrix.tobytes())
    f.close()

def maxWithIdx(l):
    max_idx = np.argmax(l)
    max_val = l[max_idx]
    return max_val, max_idx


def multiMatmul(*matrices, order='C'):
    return reduce(lambda result, mat: np.matmul(result, mat, order=order), matrices)


def svd_reconstruct(u, s, vh, order='C'):
    return np.matmul(u * s, vh, order=order)


def should_use_svds(d, k):
    """
    d - is the total s.v in the matrix (min(n,p))
    k - # of s.v we want to compute
    """
    ratio = 0.02 if d <= 100 \
        else 0.06 if d <= 200 \
        else 0.26 if d <= 300 \
        else 0.28 if d <= 400 \
        else 0.34 if d <= 500 \
        else 0.38

    return k / d <= ratio


def svd_k_largest(G, k):
    d = np.min(G.shape)
    use_svds = should_use_svds(d, k)
    if use_svds:
        u, s, vh = svds(G, k=k)
        return u[:, ::-1], s[::-1], vh[::-1, :]  # svds return s.v in descending order
    else:
        u, s, vh = LA.svd(G, full_matrices=False)
        return u[:, :k], s[0:k], vh[:k, :]


def get_last_nonzero_idx(arr):
    nonzero_elements = np.nonzero(arr)
    return np.max(nonzero_elements) if len(nonzero_elements[0]) > 0 else -1


def normalizeImage(image):
    """ Normalize image so that (min, max) -> (0, 1) """
    image -= np.min(image)
    image *= 1.0 / np.max(image)


def normalizeSparseMat(image):
    """
    takes
    """

    new_image = np.abs(image)
    new_image -= np.min(new_image)
    new_image *= 1.0 / np.max(new_image)
    binary_image = np.zeros_like(new_image)
    mask_mean = np.mean(new_image)
    mask_std = np.std(new_image)
    binary_image[new_image > mask_mean + 2*mask_std] = 1.0
    return binary_image


def output_result_bitmap_seq(folder_name, data,  lowrank_recon, sparse_mask, sparse_mask_bin):
    """
    This function takes a folder name and write frames to it
    where each frame is a concatenation of a data frame, low rank frame, sparse frame and sparse mask frame
    """
    video_data = np.zeros((data.shape[0], 4*data.shape[1], data.shape[2]), dtype=np.uint8)
    for i in range(data.shape[2]):
        data_out = (data[:, :, i]*255).astype(np.uint8)
        lowrank_out = (lowrank_recon[:, :, i]*255).astype(np.uint8)
        sparse_out = (sparse_mask[:, :, i]*255).astype(np.uint8)
        sparse_bin_out = (sparse_mask_bin[:, :, i]*255).astype(np.uint8)
        output_frame = np.concatenate((data_out, lowrank_out, sparse_out, sparse_bin_out), axis=1)
        video_data[:, :, i] = output_frame
        cv2.imwrite(folder_name+f"frame_{i}.bmp", output_frame)


def write_log_to_file(filename, args):
    now = datetime.now()
    with open(filename, "a") as logfile:
        logfile.write(f"Starting computation at {now}\n")
        for key, value in vars(args).items():
            logfile.write(f"{key} : {value}\n")
