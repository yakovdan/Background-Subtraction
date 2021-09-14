import cv2
import numpy as np
import glob
import os
from datetime import datetime
from RobustPCA.rpca import RobustPCA
from utils import bitmap_to_mat, save_images

video_length = 134  # how many frames to process
downscale_factor = 1


def compute_RPCA(image_array, name, grayscale):
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
    rpca = RobustPCA(max_iter=200000,  use_fbpca=True, max_rank=1, tol=1e-3, verbose=False)

    #start log
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    f = open(name+"_rpca_log.txt", 'w')
    f.write("Starting: "+name+" "+current_time+"\n")
    f.close()
    # perform RPCA for each image and each color channel separately
    if grayscale:
        for i in range(length):
            # compute decomposition and store
            rpca.fit(image_array[i, :, :])
            L_array[i, :, :] = rpca.get_low_rank()
            S_array[i, :, :] = rpca.get_sparse()

            # print log for this iteration
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            f = open(name+"_rpca_log.txt", 'a')
            f.write(f"Processing image {i} out of {length}.\n")
            f.write(f"Converged: {rpca.converged}, error: {rpca.error[-1]}, time: "+current_time+"\n")
            f.close()
    else:
        for i in range(length):
            for c in range(3):
                print(f"Processing image {i} out of {length}, Performing {c} th fit of 3 ")
                # compute decomposition and store
                rpca.fit(image_array[i, :, :, c])
                L_array[i, :, :, c] = rpca.get_low_rank()
                S_array[i, :, :, c] = rpca.get_sparse()

                # print log for this iteration
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                f = open(name+"_rpca_log.txt", 'a')
                f.write(f"Processing image {i} out of {length}, Performing {c} th fit of 3.\n")
                f.write(f"Converged: {rpca.converged}, error: {rpca.error[-1]}, time: "+current_time+"\n")
                f.close()

    # store output to file
    open("L_video_bin_dump_"+name+".bin", 'wb').write(L_array.tobytes())
    open("S_video_bin_dump_"+name+".bin", 'wb').write(S_array.tobytes())
    open("image_array_dump"+name+".bin", "wb").write(image_array.tobytes())

    # finish log
    f = open(name+"_rpca_log.txt", 'a')
    f.write("Finished: "+name+"\n")
    f.close()
    return L_array, S_array


def execute(grayscale=True):
    frame_names = glob.glob("./input/*.jpg")
    video_data = bitmap_to_mat(frame_names[:video_length:1])
    if grayscale:
        video_grayscale_data = np.zeros(video_data.shape[0:3], dtype=video_data.dtype)
        for i in range(video_data.shape[0]):
            video_grayscale_data[i, :, :] = cv2.cvtColor(video_data[i, :, :], cv2.COLOR_BGR2GRAY)
        video_data = video_grayscale_data
    # compute X-T and Y-T frames
    xt_plane = np.copy(video_data)
    yt_plane = np.copy(video_data)
    if grayscale:
        xt_plane = xt_plane.transpose([2, 1, 0])  # new order of axis relative to [t,h,w]
        yt_plane = yt_plane.transpose([1, 2, 0])
    else:
        xt_plane = xt_plane.transpose([2, 1, 0, 3])  # new order of axis relative to [t,h,w,c]
        yt_plane = yt_plane.transpose([1, 2, 0, 3])

    save_images(xt_plane, "output_xt", grayscale)
    save_images(yt_plane, "output_yt", grayscale)
    if grayscale:
        xt_plane = xt_plane[:, ::downscale_factor, ::downscale_factor].astype(np.float64)
        yt_plane = yt_plane[:, ::downscale_factor, ::downscale_factor].astype(np.float64)
    else:
        xt_plane = xt_plane[:, ::downscale_factor, ::downscale_factor, :].astype(np.float64)
        yt_plane = yt_plane[:, ::downscale_factor, ::downscale_factor, :].astype(np.float64)

    #compute decomposition for X-T plane
    print("Starting xt RPCA")

    xt_lowrank, xt_sparse = compute_RPCA(xt_plane, "xt_plane", grayscale)
    save_images(xt_sparse, "output_xt_sparse", grayscale)
    save_images(xt_lowrank, "output_xt_lowrank", grayscale)

    #compute decomposition for Y-T plane
    print("Starting yt RPCA")
    yt_lowrank, yt_sparse = compute_RPCA(yt_plane, "yt_plane", grayscale)
    save_images(yt_sparse, "output_yt_sparse", grayscale)
    save_images(yt_lowrank, "output_yt_lowrank", grayscale)


if __name__ == '__main__':
    os.chdir("./boats")
    print("Running in: "+str(os.getcwd()))
    execute()
