import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob

from functools import reduce
from numpy import linalg as LA
from scipy.sparse.linalg import svds
import multiprocessing
from datetime import datetime
import networkx as nx

num_cores = multiprocessing.cpu_count()
USE_PARALLEL = False


def get_usable_cores():
    return num_cores - 1


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


def import_video_as_frames(path, start, end, file_ending="jpg", work_type=np.float64):
    """
    import video
    using dtype=np.float64 to allow normalizing. use np.uint8 if not needed.
    sort frames in ascending number order because glob can return filenames in any order

    video_data is in [t,h,w] order and is a c-order array
    ImData0 is in [h,w, t] order and is a fortran array

    """
    frames_list = glob.glob(path + '*.'+file_ending)
    frames_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    frames_list = frames_list[start:end+1]
    print(f"first to load: {frames_list[0]}, last to load: {frames_list[-1]}")
    bitmap_seq = bitmap_to_mat(frames_list, grayscale=True).astype(work_type)
    video_data = np.ascontiguousarray(bitmap_seq.transpose((2, 0, 1)))
    ImData0 = np.asfortranarray(bitmap_seq)
    print ("done loading")
    return ImData0, video_data


def save_images(image_array, path, grayscale):
    """
    given an video array and a folder path, save each image frame to file
    by iterating over the first axis of the array
    """

    num_images = image_array.shape[0]
    filenames = [path + f"/output_image{i}.bmp" for i in range(num_images)]
    for i in range(num_images):
        if grayscale:
            cv2.imwrite(filenames[i], image_array[i, :, :])
        else:
            cv2.imwrite(filenames[i], image_array[i, :, :, :])


def plot_errors(errors_list, filename, title, x_label, y_label, display=False, log_scale=True):
    iterations = np.arange(1, len(errors_list) + 1)
    if log_scale:
        data = np.log(np.array(errors_list))
    else:
        data = np.array(errors_list)
    plt.plot(iterations, data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)
    if display:
        plt.show()
    plt.clf()

def resize_with_cv2_timefirst(images, ratio):
    result_size = [int(np.ceil(images.shape[i] * ratio)) for i in [1, 2]]
    T = images.shape[0]
    result = np.empty([T] + result_size)
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


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def sub2indF(i, j, array_shape):
    return j*array_shape[0] + i


def get_vars_idx_center(i, j, group_radius, img_shape):
    rows, cols = img_shape

    left = min(group_radius, j)
    right = min(group_radius, cols - 1 - j)
    top = min(group_radius, i)
    bottom = min(group_radius, rows - 1 - i)

    top_left_idx = sub2indF(i - top, j - left, img_shape)

    horiz_width = left + right + 1
    vert_width = top + bottom + 1
    return [top_left_idx + di + rows * dj for dj in range(horiz_width) for di in range(vert_width)]


def get_vars_idx_top_left(i, j, group_shape, img_shape):
    rows, cols = img_shape

    bottom = min(group_shape[0], rows - 1 - i)
    right = min(group_shape[1], cols - 1 - j)

    top_left_idx = sub2indF(i, j, img_shape)

    return [top_left_idx + di + rows * dj for dj in range(right) for di in range(bottom)]


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
    binary_image[new_image > mask_mean + 2 * mask_std] = 1.0
    return binary_image


def output_result_bitmap_seq(folder_name, data, lowrank_recon, sparse_mask, sparse_mask_bin):
    """
    This function takes a folder name and write frames to it
    where each frame is a concatenation of a data frame, low rank frame, sparse frame and sparse mask frame
    """
    video_data = np.zeros((data.shape[0], 4 * data.shape[1], data.shape[2]), dtype=np.uint8)
    for i in range(data.shape[2]):
        data_out = (data[:, :, i] * 255).astype(np.uint8)
        lowrank_out = (lowrank_recon[:, :, i] * 255).astype(np.uint8)
        sparse_out = (sparse_mask[:, :, i] * 255).astype(np.uint8)
        sparse_bin_out = sparse_mask_bin[:, :, i].astype(np.uint8) * 255
        output_frame = np.concatenate((data_out, lowrank_out, sparse_out, sparse_bin_out), axis=1)
        video_data[:, :, i] = output_frame
        cv2.imwrite(folder_name + f"frame_{i}.bmp", output_frame)
    np.save(folder_name+"video_data_out", video_data)

def write_log_to_file(filename, args):
    now = datetime.now()
    with open(filename, "a") as logfile:
        logfile.write(f"Starting computation at {now}\n")
        for key, value in vars(args).items():
            logfile.write(f"{key} : {value}\n")


def print_to_logfile(filename, msg):
    with open(filename, 'a') as f:
        f.write(msg+'\n')


def parse_numerical_values(path):
    with open(path, 'r') as vals_file:
        line = vals_file.read()
        params = line.split(",")
        first_val_str = params[0].split(":")[1]
        second_string = " ".join(params[1:])
        second_params = second_string.split(":")
        second_val_str = second_params[1]
        video_mean = float(first_val_str)
        second_val_str = second_val_str.lstrip()
        second_val_str = second_val_str.rstrip()
        second_val_str = second_val_str[1:-1]
        second_val_Lst = second_val_str.split("\n")
        second_val_str = second_val_Lst[0]
        second_val_str = second_val_str[:-1]
        second_val_Lst = second_val_str.split(" ")
        int_vals = [int(x) for x in second_val_Lst if len(x) > 0]
        original_downsampled_shape = tuple(int_vals)
    return video_mean, original_downsampled_shape


def show_sequence_on_screen(np_array, use_abs=False, video_mean=0, indices=None):
    min_val = np.min(np_array)
    max_val = np.max(np_array)
    shape = np_array.shape
    if indices is None:
        indices = list(range(0, shape[2], shape[2] // 10))
    for i in indices:
        if use_abs:
            frame = np.abs(np_array[:, :, i])
        else:
            frame = np_array[:, :, i]

        cv2.imwrite('test.bmp', (255 * (frame + video_mean)).astype(np.uint8))
        cv2.imshow('OnScreen', frame + video_mean)
        cv2.waitKey(0)


def contained_in(cc1, cc2):
    """
        checks if the rectangle [x1,y1,w1,h1] is contained in [x2,y2,w2,h2]
    """
    x2, y2, w2, h2 = cc2
    x1, y1, w1, h1 = cc1
    if x2 < x1 and y2 < y1 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2:
        return True

    return False


def unite_nestedCCs(num_labels, labels, stats):
    """
    given a number of labels, the labels np array and stats from cv2.connectedcomponents
    this function returns a new labels array and a dictionary of labels and areas
    such that all nested CCs have the same label and count towards the same area

    """

    cc_dict = {}
    nested_cc_list = []  # (1, 2) is in nested_cc_list iff cc 1's bbox is contained in cc 2's bbox
    for i in range(0, num_labels):

        # i ==0 is the background cc by convention. ignore
        if i == 0:
            continue

        # extract stats per cc
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cc_dict[i] = ((x, y, w, h), area)

    for label1, stats1 in cc_dict.items():
        for label2, stats2 in cc_dict.items():
            if label1 == label2:
                continue
            if contained_in(stats1[0], stats2[0]):
                nested_cc_list.append((label1, label2))

    nested_reverse_cc_list = [(x[1], x[0]) for x in nested_cc_list]
    graph = nx.Graph()

    for n1, n2 in nested_reverse_cc_list:
        graph.add_edge(n1, n2)

    spanning_tree = nx.minimum_spanning_tree(graph)
    edges = spanning_tree.edges()
    new_labels = np.copy(labels)

    for n1, n2 in edges:
        new_labels[labels == n2] = n1

    unique, counts = np.unique(new_labels, return_counts=True)
    ccs_dict = dict(zip(unique, counts))
    del ccs_dict[0]
    return ccs_dict, new_labels


def filter_sparse_map(sparse_array, size_thresh=None):
    """
    This function takes a binary sparse array
    and filters it such that no object smaller than size_thresh remains
    """
    connectivity = 8
    if size_thresh is None:
        size_thresh = (sparse_array.shape[0] * sparse_array.shape[1]) // 200  # from paper
    result_sparse_array = np.zeros_like(sparse_array)

    for i in range(sparse_array.shape[2]):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sparse_array[:, :, i].astype(np.uint8)*255,
                                                                                connectivity, cv2.CV_32S)
        for j in range(1, num_labels):
            if stats[j, cv2.CC_STAT_AREA] > size_thresh:
                result_sparse_array[:, :, i][labels == j] = True
    return result_sparse_array


def find_bounding_area_in_frame(sparse_frame):
    connectivity = 4
    work_frame = np.copy(sparse_frame)
    if work_frame.dtype != np.uint8:
        work_frame = work_frame.astype(np.uint8)

    work_frame[work_frame != 0] = 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(work_frame, connectivity, cv2.CV_32S)
    cc_dimensions = []
    for i in range(0, num_labels):
        # i ==0 is the background cc by convention. ignore
        if i == 0:
            continue
        cc_dimensions.append((stats[i, cv2.CC_STAT_LEFT],
                              stats[i, cv2.CC_STAT_TOP],
                              stats[i, cv2.CC_STAT_WIDTH]+stats[i, cv2.CC_STAT_LEFT],
                              stats[i, cv2.CC_STAT_HEIGHT]+stats[i, cv2.CC_STAT_TOP]))

    cc_dimensions.sort(key=lambda x: x[0])
    min_left = cc_dimensions[0][0]
    cc_dimensions.sort(key=lambda x: x[1])
    min_top = cc_dimensions[0][1]
    cc_dimensions.sort(key=lambda x: x[2])
    max_right = cc_dimensions[-1][2]
    cc_dimensions.sort(key=lambda x: x[3])
    max_bottom = cc_dimensions[-1][3]

    return min_left, min_top, max_right, max_bottom


def find_bounding_volume(sparse_mat):
    bounding_boxes = [find_bounding_area_in_frame(sparse_mat[:, :, i]) for i in sparse_mat.shape[2]]
    bounding_boxes.sort(key=lambda x: x[0])
    min_left = bounding_boxes[0][0]
    bounding_boxes.sort(key=lambda x: x[1])
    min_top = bounding_boxes[0][1]
    bounding_boxes.sort(key=lambda x: x[2])
    max_right = bounding_boxes[-1][2]
    bounding_boxes.sort(key=lambda x: x[3])
    max_bottom = bounding_boxes[-1][3]
    return min_left, min_top, max_right, max_bottom

