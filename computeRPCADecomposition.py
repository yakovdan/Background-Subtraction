import cv2
import numpy as np
import glob
import os
from datetime import datetime
from RobustPCA.rpca import RobustPCA


video_length = 100  # how many frames to process
downscale_factor = 1

def save_images(image_array, path):
    """
    given an video array and a foler path, save each image frame to file
    by iterating over the first axis of the array
    """

    num_images = image_array.shape[0]
    filenames = [path+f"/output_image{i}.bmp" for i in range(num_images)]
    for i in range(num_images):
        cv2.imwrite(filenames[i], image_array[i, :, :, :])


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


def execute():
    video_data = bitmap_to_mat(glob.glob("./input/*.jpg")[:video_length:1])

    # compute X-T and Y-T frames
    xt_plane = np.copy(video_data)
    xt_plane = xt_plane.transpose([2, 1, 0, 3])  # new order of axis relative to [t,h,w,c]
    yt_plane = np.copy(video_data)
    yt_plane = yt_plane.transpose([1, 2, 0, 3])
    save_images(xt_plane, "output_xt")
    save_images(xt_plane, "output_yt")

    xt_plane = xt_plane[:, ::downscale_factor, ::downscale_factor, :].astype(np.float64)
    yt_plane = yt_plane[:, ::downscale_factor, ::downscale_factor, :].astype(np.float64)

    #compute decomposition for X-T plane
    print("Starting xt RPCA")
    xt_lowrank, xt_sparse = compute_RPCA(xt_plane, "xt_plane")
    save_images(xt_sparse, "output_xt_sparse")
    save_images(xt_lowrank, "output_xt_lowrank")

    #compute decomposition for Y-T plane
    print("Starting yt RPCA")
    yt_lowrank, yt_sparse = compute_RPCA(yt_plane, "yt_plane")
    save_images(yt_sparse, "output_yt_sparse")
    save_images(yt_lowrank, "output_yt_lowrank")




if __name__ == '__main__':
    print("Running in: "+str(os.getcwd()))
    execute()
