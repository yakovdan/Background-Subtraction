import cv2
import glob
import numpy as np
from utils import *


def blend(input_video_array, input_mask_array):

    blend_array = np.copy(input_video_array[:, :, :, :])
    blend_array[(input_mask_array[:, :, :] != 255), :] = 0
    temp_mask_array = np.stack((input_mask_array, input_mask_array, input_mask_array), axis=-1)
    output_array = np.concatenate((input_video_array[:, :, :, :], temp_mask_array, blend_array), axis=2)
    return output_array


if __name__ == '__main__':
    video_list = glob.glob("./input/*.jpg")
    video_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    mask_list = glob.glob("./binary_video/*.bmp")
    mask_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    video_array = bitmap_to_mat(video_list)
    mask_array = bitmap_to_mat(mask_list, grayscale=True)
    video_length = min(video_array.shape[0], mask_array.shape[0])
    video_array = video_array[0:video_length, :, :, :]  # color video
    mask_array = mask_array[0:video_length, :, :]  # binary video, no color channel
    blended_array = blend(video_array, mask_array)
    save_images(blended_array, "blend_test", grayscale=True)
