import os
import cv2
from PIL import Image as im
import numpy as np
from scipy.ndimage import interpolation
import matplotlib.pyplot as plt


def crop_images(input_path, output_path):
    for image in os.listdir(input_path):
        if image.endswith('.jpg') or image.endswith('.png'):
            img = cv2.imread(os.path.join(input_path, image))
            img = cv2.resize(img, (1042, 1464))

            # Change if the format of the page is different
            mask_even_page = (slice(100, 1205), slice(120, 900))
            mask_odd_page = (slice(95, 1195), slice(180, 970))

            crop_img = img[mask_even_page] if (int(image[-5]) % 2 == 0) else img[mask_odd_page]
            cv2.imwrite(os.path.join(output_path, f"{image[:-4]}.png"), crop_img)


def deskew_images(input_path, output_path):
    for file_name in sorted(os.listdir(input_path)):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            img = im.open(os.path.join(input_path, file_name))
            name_img = os.path.basename(file_name)[:-4]

            # convert to binary
            width, height = img.size
            pix = np.array(img.convert('1').getdata(), np.uint8)
            bin_img = 1 - (pix.reshape((height, width)) / 255.0)

            delta = 1
            limit = 5
            angles = np.arange(-limit, limit + delta, delta)
            scores = []
            for angle in angles:
                hist, score = find_score(bin_img, angle)
                scores.append(score)

            best_score = max(scores)
            best_angle = angles[scores.index(best_score)]
            print(f'Best angle: {best_angle} {name_img}')

            # correct skew
            data = interpolation.rotate(img, best_angle, reshape=False, order=0, mode='nearest')
            img = im.fromarray((data).astype("uint8")).convert("RGB")
            img.save(os.path.join(os.path.abspath(output_path), f"{name_img}.png"))


def find_score(arr, angle):
    data = interpolation.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score
