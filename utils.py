import cv2
import numpy as np
import matplotlib.pyplot as plt

def bitmap_to_mat(bitmap_seq, grayscale=False):
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
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if shape is None:  # first image read
            shape = img.shape
            if grayscale:
                matrix = np.zeros((image_count, shape[0], shape[1]), dtype=np.uint8)
            else:
                matrix = np.zeros((image_count, shape[0], shape[1], shape[2]), dtype=np.uint8)
        assert img.shape == shape
        if grayscale:
            matrix[count, :, :] = img
        else:
            matrix[count, :, :, :] = img
        count = count + 1
    return matrix


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