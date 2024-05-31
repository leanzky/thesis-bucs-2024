import cv2
import numpy as np
import math
import sys
import os
import argparse

# Interpolation kernel
def u(s, a):

    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s) ** 3) - (5 * a) * (abs(s) ** 2) + (8 * a) * abs(s) - 4 * a
    return 0

# Bicubic operation
def bicubic(img, ratio, a):
    # Get image size
    H, W, C = img.shape

    # Create new image
    dH = 128
    dW = 128
    dst = np.zeros((dH, dW, C))

    h = 1 / ratio

    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.matrix([[img[int(y - y1), int(x - x1), c], img[int(y - y2), int(x - x1), c], img[int(y + y3), int(x - x1), c], img[int(y + y4), int(x - x1), c]], [img[int(y - y1), int(x - x2), c], img[int(y - y2), int(x - x2), c], [int(y + y3), int(x - x2), c], img[int(y + y4), int(x - x2), c]], [img[int(y - y1), int(x + x3), c], img[int(y - y2), int(x + x3), c], img[int(y + y3), int(x + x3), c], img[int(y + y4), int(x + x3), c]], [img[int(y - y1), int(x + x4), c], img[int(y - y2), int(x + x4), c], img[int(y + y3), int(x + x4), c], img[int(y + y4), int(x + x4), c]]])
                mat_r = np.matrix([[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

    return dst

# Create an argument parser
parser = argparse.ArgumentParser(description='Bicubic interpolation script')

# Add arguments for folder paths
parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder')
parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder')

# Parse the arguments
args = parser.parse_args()

# Use the parsed arguments
folder_path = args.input_folder
new_folder_path = args.output_folder

# Create the new folder if it doesn't exist
os.makedirs(new_folder_path, exist_ok=True)

# Scale factor
ratio = 2
# Coefficient
a = -1 / 2

# Iterate over files in the folder
for filename in os.listdir(folder_path):
   # Check if the file is an image
   if filename.endswith('.jpg') or filename.endswith('.png'):
       # Construct the full file path
       file_path = os.path.join(folder_path, filename)

       # Read the image
       img = cv2.imread(file_path)

       # Apply bicubic interpolation
       interpolated_img = bicubic(img, ratio, a)

       # Resize the interpolated image to 128x128
       resized_img = cv2.resize(interpolated_img, (128, 128), interpolation=cv2.INTER_CUBIC)

       # Save the resized image with a new filename in the new folder
       resized_filename = f'{filename}'
       resized_file_path = os.path.join(new_folder_path, resized_filename)
       cv2.imwrite(resized_file_path, resized_img)

print('Completed!')