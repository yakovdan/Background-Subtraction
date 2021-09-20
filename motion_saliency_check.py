import numpy as np
from utils import *
import cv2


def compute_lsd_mask(data, lowrank_mat, sparse_mat):
    """
    this function takes a data, low rank and sparse matricesx
    and returns a binary mask for the sparse matrix

    data should be normalized and centered
    """

    size_thresh = (data.shape[0] * data.shape[1]) / 1500
    mask = foreground_mask(data, lowrank_mat, sparse_mat, sigmas_from_mean=2)
    mask_image = (mask*255).astype(np.uint8)
    return mask_image, size_thresh


def compute_groups_per_frame(mask_image, sparse_cube, frame_idx):
    """
    this function takes a binary mask, a size threshold, a frame_idx
    and a sparse cube and computes groups and weights for the specified frame
    """

    groups = []
    connectivity = 4  # You need to choose 4 or 8 for connectivity type
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_image[:, :, frame_idx],
                                                                            connectivity, cv2.CV_32S)
    areas = {}
    for i in range(0, num_labels):
        # i ==0 is the background cc by convention. ignore
        if i == 0:
            continue

        #extract stats per cc
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        (cX, cY) = centroids[i]
        area = stats[i, cv2.CC_STAT_AREA]
        areas[i] = area

    for label, area in areas.items():
        mask_2d = labels == label
        mask_1d = mask_2d.flatten(order='F')
        weight = np.sum(sparse_cube[:, :, frame_idx][mask_2d]) / area
        groups.append((frame_idx, weight, area, mask_1d))

    return groups


def filter_groups(groups, size_thresh):
    """
    this function takes a list of group tuples in the format (frame_id, weight, area, binary_map)
    and returns a list of groups filtered by weight and size and the minimum weight
    """
    all_weights = np.array([g[1] for g in groups])
    mean_weight = np.mean(all_weights)
    std_weight = np.std(all_weights)
    threshold_weight = mean_weight + std_weight
    weight_filtered_groups = list(filter(lambda g: g[1] > threshold_weight, groups))  # g[1] is the group weight
    size_filtered_groups = list(filter(lambda g: g[2] > size_thresh, weight_filtered_groups))  #g[2] is the group area
    return size_filtered_groups, min([g[1] for g in size_filtered_groups])


def run_motion_saliency_check(data, lowrank_mat, sparse_mat, sparse_cube):
    """
    this function takes a data matrix: np array of the video in [w,h,t] format, np.float64
    lowrank_mat - lowrank output of lsd, same format as above
    sparse_mat - sparse output of lsd, same format as above
    sparse_cube - output of saliency step, same format as above
    """

    all_groups = []
    groups_by_frame = []
    weights_by_frame = []
    shape = data.shape
    video_length = shape[2]

    #########################################
    # compute foreground mask from lsd step.
    # compute size threshold for filtering.
    #########################################

    lsd_mask, size_thresh = compute_lsd_mask(data, lowrank_mat, sparse_mat)

    ###############################
    # compute groups for each frame
    ################################

    for i in range(video_length):
        groups_of_frame = compute_groups_per_frame(lsd_mask, sparse_cube, i)
        all_groups.extend(groups_of_frame)

    ##################################################
    # filter groups using an adaptive weight threshold
    # and a size threshold
    ##################################################

    all_filtered_groups, min_weight = filter_groups(all_groups, size_thresh)
    all_filtered_groups.sort(key=lambda g: g[0])  # sort by ascending frame idx

    ###################################################
    # compute normalization factor to convert each weight
    # to lambda_i value from the paper
    ########################################

    delta = 10
    normalization_factor_for_lambda = 1.0/(delta * max(shape[0]*shape[1], shape[2]))
    normalization_factor_for_lambda = normalization_factor_for_lambda * min_weight

    ############################################
    # convert flat list into two lists of group
    # masks and lambda_i weights
    ############################################

    for frame_idx in range(video_length):
        filter_groups_by_frame = list(filter(lambda g: g[0] == frame_idx, all_filtered_groups))
        groups_by_frame.append([g[3] for g in filter_groups_by_frame])
        weights_by_frame.append([(normalization_factor_for_lambda / g[1]) for g in filter_groups_by_frame])

    return groups_by_frame, weights_by_frame
