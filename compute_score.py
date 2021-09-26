import argparse
import sys
import time

import cv2
import numpy as np
from utils import *


def create_pretty_score_map(sparse_mat, gt_mat):
    pretty_map = np.zeros((list(sparse_mat.shape)+[3]), dtype=np.uint8)
    map_tp = np.logical_and(sparse_mat, gt_mat)
    map_fp = np.logical_and(sparse_mat, np.logical_not(gt_mat))
    map_fn = np.logical_and(np.logical_not(sparse_mat), gt_mat)
    # place a white pixel for TP
    pretty_map[map_tp, :] = np.array((255, 255, 255), dtype=np.uint8)
    # place a red pixel for FN
    pretty_map[map_fn, :] = np.array((0, 0, 255), dtype=np.uint8)
    # place a blue pixel for FP
    pretty_map[map_fp, :] = np.array((255, 0, 0), dtype=np.uint8)
    return pretty_map


# gt says it's an object, spare result agrees:
def true_positive(sparse_mat, gt_mat):
    tp_list = []
    for i in range(sparse_mat.shape[2]):
        sparse_frame = sparse_mat[:, :, i]
        gt_frame = gt_mat[:, :, i]
        tp_list.append(np.sum(sparse_frame[gt_frame == 1]))
    return np.array(tp_list)


# gt says it's background but sparse result says it's an object
def false_positive(sparse_mat, gt_mat):
    fp_list = []
    for i in range(sparse_mat.shape[2]):
        sparse_frame = sparse_mat[:, :, i]
        gt_frame = gt_mat[:, :, i]
        fp_list.append(np.sum(sparse_frame[gt_frame == 0]))
    return np.array(fp_list)


# gt says it's an object but sparse result says it's background
def false_negative(sparse_mat, gt_mat):
    fn_list = []
    for i in range(sparse_mat.shape[2]):
        sparse_frame = sparse_mat[:, :, i]
        gt_frame = gt_mat[:, :, i]
        fn_list.append(np.sum(gt_frame[sparse_frame == 0]))
    return np.array(fn_list)


# TP / (TP+FP)
def compute_precision(tp_list, fp_list):
    return tp_list / (tp_list+fp_list)


# TP / (TP+FN)
def compute_recall(tp_list, fn_list):
    return tp_list / (tp_list+fn_list)


def compute_fscore(tp_list, fp_list, fn_list):
    rc = compute_recall(tp_list, fn_list)
    pr = compute_precision(tp_list, fp_list)
    return 2*rc*pr/(rc+pr)


def read_gt_start_stop_frames(path):
    with open(path + 'temporalROI.txt', 'r') as timedata:
        line = timedata.read()
        vals = tuple([int(x) for x in line.split()])
        return vals


def main(args):
    start_gt_frame, end_gt_frame = read_gt_start_stop_frames(args.input)
    gt_frames, _ = import_video_as_frames(args.input + "/groundtruth/", start_gt_frame - 1, end_gt_frame,
                                          file_ending="png", work_type=np.uint8)
    gt_frames = np.ascontiguousarray(gt_frames)

    if args.discard_segmentation:
        gt_frames[gt_frames != 0] = 255
    gt_frames = gt_frames.astype(np.bool_)
    sparse_mat = np.load(args.sparse)[:, :, start_gt_frame - 1:].astype(np.float64)
    if sparse_mat.shape[:2] != gt_frames.shape[:2]:  # not the same scale
        height_scale = gt_frames.shape[0] // sparse_mat.shape[0]
        width_scale = gt_frames.shape[1] // sparse_mat.shape[1]

        if height_scale != width_scale:
            print("cant resize sparse matrix to match gt and keep the same aspect ratio. something went wrong!")
            sys.exit(-1)

        sparse_mat_resize = np.zeros((sparse_mat.shape[0] * height_scale,
                                      sparse_mat.shape[1] * width_scale,
                                      sparse_mat.shape[2]), dtype=sparse_mat.dtype)

        for i in range(sparse_mat.shape[2]):
            sparse_mat_resize[:, :, i] = cv2.resize(src=sparse_mat[:, :, i],
                                                    dsize=(sparse_mat.shape[1] * width_scale,
                                                           sparse_mat.shape[0] * height_scale),
                                                    interpolation=cv2.INTER_AREA)
        sparse_mat = sparse_mat_resize

    sparse_mat = sparse_mat > 0
    assert sparse_mat.shape, gt_frames.shape
    assert sparse_mat.dtype, gt_frames.dtype

    tp_array = true_positive(sparse_mat, gt_frames)
    fp_array = false_positive(sparse_mat, gt_frames)
    fn_array = false_negative(sparse_mat, gt_frames)

    precision_array = compute_precision(tp_array, fp_array)
    recall_array = compute_recall(tp_array, fn_array)
    fscore_array = compute_fscore(tp_array, fp_array, fn_array)

    plot_errors(precision_array, args.output+"precision.png",
                "Precision over frames", "frames", "precision",
                display=True,
                log_scale=False)

    plot_errors(recall_array, args.output+"recall.png",
                "Recall over frames", "frames", "recall",
                display=True,
                log_scale=False)

    plot_errors(fscore_array, args.output+"fscore.png",
                "Fscore over frames", "frames", "fscore",
                display=True,
                log_scale=False)

    if args.output_video:
        pretty_mat = create_pretty_score_map(sparse_mat, gt_frames)
        np.save(args.output+"pretty_mat.bin", pretty_mat)
    np.save(args.output+"tp_array", tp_array)
    np.save(args.output+"fp_array", fp_array)
    np.save(args.output+"fn_array", fn_array)

    np.save(args.output+"precision_array", precision_array)
    np.save(args.output+"reall", recall_array)
    np.save(args.output+"fscore", fscore_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run LSD')
    parser.add_argument('--input', type=str, default=".", help='path to dataset root folder')
    parser.add_argument('--output', type=str, default=".", help='path to dataset root folder')
    parser.add_argument('--sparse', type=str, default=".", help='path to sparse matrix file')
    parser.add_argument('--discard_segmentation', type=bool, default=True,
                        help='keep or discard semantic values in GT images')
    parser.add_argument('--output_video', type=bool, default=False, help="output video or not")
    args = parser.parse_args()

    print('START')
    write_log_to_file(args.output + 'gtlog.txt', args)
    start = time.time()
    main(args)
    end = time.time()
    print('DONE')
    print(f'ELAPSED TIME: {(end - start):.3f} seconds')
    with open(args.output + 'gtlog.txt', 'a') as f:
        f.write(f'ELAPSED TIME: {(end - start):.3f} seconds')
