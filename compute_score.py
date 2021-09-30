import argparse
import time
from utils import *

known_values = [0, 50, 255]  #  from CDNET 2014

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
def true_positive(sparse_mat, gt_mat, roi_mask):
    tp_list = []
    roi_mask_bool = roi_mask == 255
    for i in range(sparse_mat.shape[2]):
        sparse_frame = sparse_mat[:, :, i]
        gt_frame = gt_mat[:, :, i]
        gt_search_area = np.logical_and(np.isin(gt_frame, known_values), roi_mask_bool)
        gt_frame_bool = np.logical_and(gt_search_area, gt_frame == 255)
        tp_frame_bool = np.logical_and(gt_frame_bool, sparse_frame)
        tp_list.append(np.sum(tp_frame_bool))
    return np.array(tp_list)


# gt says it's background but sparse result says it's an object
def false_positive(sparse_mat, gt_mat, roi_mask):
    fp_list = []
    roi_mask_bool = roi_mask == 255
    for i in range(sparse_mat.shape[2]):
        sparse_frame = sparse_mat[:, :, i]
        gt_frame = gt_mat[:, :, i]
        gt_search_area = np.logical_and(np.isin(gt_frame, known_values), roi_mask_bool)
        gt_frame_bool = np.logical_and(gt_search_area, gt_frame != 255)  # not an object
        fp_frame_bool = np.logical_and(gt_frame_bool, sparse_frame) # sparse says it's an object
        fp_list.append(np.sum(fp_frame_bool))
    return np.array(fp_list)


# gt says it's an object but sparse result says it's background
def false_negative(sparse_mat, gt_mat, roi_mask):
    fn_list = []
    roi_mask_bool = roi_mask == 255
    for i in range(sparse_mat.shape[2]):
        sparse_frame = sparse_mat[:, :, i]
        gt_frame = gt_mat[:, :, i]
        gt_search_area = np.logical_and(np.isin(gt_frame, known_values), roi_mask_bool)
        gt_frame_bool = np.logical_and(gt_search_area, gt_frame == 255)  # an object!
        fn_frame_bool = np.logical_and(gt_frame_bool, np.logical_not(sparse_frame))  # sparse says it's not an object
        fn_list.append(np.sum(fn_frame_bool))
    return np.array(fn_list)


# TP / (TP+FP)
def compute_precision(tp_list, fp_list):
    result = np.zeros(tp_list.shape, np.float32)
    for i in range(tp_list.size):
        if tp_list[i] == 0 and fp_list[i] == 0:
            result[i] = 1
        else:
            result[i] = tp_list[i] / (tp_list[i]+fp_list[i])

    return result


# TP / (TP+FN)
def compute_recall(tp_list, fn_list):
    result = np.zeros(tp_list.shape, np.float32)
    for i in range(tp_list.size):
        if tp_list[i] == 0 and fn_list[i] == 0:
            result[i] = 1
        else:
            result[i] = tp_list[i] / (tp_list[i]+fn_list[i])

    return result


def compute_fscore(tp_list, fp_list, fn_list):
    rc = compute_recall(tp_list, fn_list)
    pr = compute_precision(tp_list, fp_list)
    result = np.zeros(rc.shape, np.float32)
    for i in range(rc.size):
        if rc[i] == 0 and pr[i] == 0:
            result[i] = 1
        else:
            result[i] = 2*rc[i]*pr[i] / (rc[i]+pr[i])

    return result


def read_gt_start_stop_frames(path):
    with open(path + 'temporalROI.txt', 'r') as timedata:
        line = timedata.read()
        vals = tuple([int(x) for x in line.split()])
        return vals



def main(args):
    start_gt_frame, end_gt_frame = read_gt_start_stop_frames(args.input)
    start_gt_frame = max(start_gt_frame, args.start_gt_ind)
    roi_mask = cv2.cvtColor(cv2.imread(args.input+"ROI.bmp"), cv2.COLOR_BGR2GRAY)

    gt_frames, _ = import_video_as_frames(args.input + "/groundtruth/",
                                          start_gt_frame - 1,
                                          end_gt_frame,
                                          file_ending="png",
                                          work_type=np.uint8)

    gt_frames = np.ascontiguousarray(gt_frames)
    sparse_mat = np.load(args.sparse)[:, :, args.start_ind:]
    print(f"sparse mat dtype {sparse_mat.dtype}")
    if sparse_mat.shape[:2] != gt_frames.shape[:2]:  # not the same scale
        height_scale = gt_frames.shape[0] // sparse_mat.shape[0]
        width_scale = gt_frames.shape[1] // sparse_mat.shape[1]
        print(f"resizing: {height_scale} {width_scale}")
        if height_scale != width_scale:
            print("cant resize sparse matrix to match gt and keep the same aspect ratio. something went wrong!")
            raise Exception("Can't resize while keeping aspect ratio")

        sparse_mat_resize = np.zeros((sparse_mat.shape[0] * height_scale,
                                      sparse_mat.shape[1] * width_scale,
                                      sparse_mat.shape[2]), dtype=sparse_mat.dtype)

        for i in range(sparse_mat.shape[2]):
            sparse_mat_resize[:, :, i] = np.kron(sparse_mat[:, :, i], np.ones((height_scale, height_scale), bool))
        sparse_mat = sparse_mat_resize

    print(sparse_mat.shape, gt_frames.shape)
    assert sparse_mat.shape == gt_frames.shape

    tp_array = true_positive(sparse_mat, gt_frames, roi_mask)
    fp_array = false_positive(sparse_mat, gt_frames, roi_mask)
    fn_array = false_negative(sparse_mat, gt_frames, roi_mask)

    precision_array = compute_precision(tp_array, fp_array)
    recall_array = compute_recall(tp_array, fn_array)
    fscore_array = compute_fscore(tp_array, fp_array, fn_array)
    print(f"Average fscore: {np.mean(fscore_array)}")
    print(f"Average recall: {np.mean(recall_array)}")
    print(f"Average precision: {np.mean(precision_array)}")

    with open(args.output+"scoredata.txt","w") as scorelog:
        scorelog.write(f"Average Fscore: {np.mean(fscore_array)}\n")
        scorelog.write(f"Average Recall: {np.mean(recall_array)}\n")
        scorelog.write(f"Average Precision: {np.mean(precision_array)}\n")

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
    np.save(args.output+"rceall", recall_array)
    np.save(args.output+"fscore", fscore_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run score calculation')
    parser.add_argument('--input', type=str, default=".", help='path to gt root folder')
    parser.add_argument('--output', type=str, default=".", help='path to output root folder')
    parser.add_argument('--sparse', type=str, default=".", help='path to sparse matrix file')
    parser.add_argument('--discard_segmentation', type=bool, default=True,
                        help='keep or discard semantic values in GT images')
    parser.add_argument('--output_video', type=bool, default=False, help="output video or not")
    parser.add_argument('--start_ind', type=int, default=0, help="first frame index of sparse mat")
    parser.add_argument('--start_gt_ind', type=int, default=0, help="first frame index of gt mat")
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
