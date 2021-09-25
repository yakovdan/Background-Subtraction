import argparse
import sys
import time

import numpy as np
from utils import *

def read_gt_start_stop_frames(path):
    with open(path+'temporalROI.txt','r') as timedata:
        line = timedata.read()
        vals = tuple([int(x) for x in line.split()])
        return vals


def main(args):
    start_gt_frame, end_gt_frame = read_gt_start_stop_frames(args.input)
    gt_frames, _ = import_video_as_frames(args.input+"/groundtruth/", start_gt_frame-1, end_gt_frame, file_ending="png", work_type=np.uint8)
    gt_frames = np.ascontiguousarray(gt_frames)

    if args.discard_segmentation:
        gt_frames[gt_frames != 0] = 255
    gt_frames = gt_frames.astype(np.bool_)
    sparse_mat = np.load(args.sparse)[:, :, start_gt_frame-1:].astype(np.float64)


    print (sparse_mat.shape, gt_frames.shape)
    print (sparse_mat.dtype, gt_frames.dtype)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run LSD')
    parser.add_argument('--input', type=str, default=".", help='path to dataset root folder')
    parser.add_argument('--output', type=str, default=".", help='path to dataset root folder')
    parser.add_argument('--sparse', type=str, default=".", help='path to sparse matrix file')
    parser.add_argument('--discard_segmentation', type=bool, default=True, help='keep or discard semantic values in GT images')
    args = parser.parse_args()

    print('START')
    write_log_to_file(args.output+'gtlog.txt', args)
    start = time.time()
    main(args)
    end = time.time()
    print('DONE')
    print(f'ELAPSED TIME: {(end - start):.3f} seconds')
    with open(args.output+'gtlog.txt', 'a') as f:
        f.write(f'ELAPSED TIME: {(end - start):.3f} seconds')
