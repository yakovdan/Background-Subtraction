from utils import *
import numpy as np

source_path = r"D:\Masters\ProcessedData\output\canoe_1189_original\score\pretty_mat.bin.npy"
output_path = r"D:\Masters\ProcessedData\output\canoe_1189_original\score\video\\"
prettymap = np.load(source_path)
print(prettymap.dtype)
for i in range (prettymap.shape[2]):
    cv2.imwrite(output_path+f"frame_{i}.bmp", prettymap[:, :, i, :])
print(prettymap.shape)
