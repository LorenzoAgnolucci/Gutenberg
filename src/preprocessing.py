import os
import pathlib

import cv2
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation


def crop_images(input_path, output_path):
    for file_name in os.listdir(input_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image = cv2.imread(os.path.join(input_path, file_name))
            image = cv2.resize(image, (1042, 1464))

            # Change if the format of the page is different
            mask_even_page = (slice(100, 1205), slice(120, 900))
            mask_odd_page = (slice(95, 1195), slice(180, 970))

            crop_image = image[mask_even_page] if (int(file_name[-5]) % 2 == 0) else image[mask_odd_page]
            cv2.imwrite(os.path.join(output_path, f"{file_name[:-4]}.png"), crop_image)


def deskew_images(input_path, output_path):
    for file_name in sorted(os.listdir(input_path)):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image = im.open(os.path.join(input_path, file_name))
            output_file = os.path.basename(file_name)[:-4]

            # convert to binary
            width, height = image.size
            pix = np.array(image.convert('1').getdata(), np.uint8)
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
            print(f'Best angle: {best_angle} {file_name}')

            # correct skew
            data = interpolation.rotate(image, best_angle, reshape=False, order=0, mode='nearest')
            image = im.fromarray(data.astype("uint8")).convert("RGB")
            image.save(os.path.join(os.path.abspath(output_path), f"{output_file}.png"))


def find_score(arr, angle):
    data = interpolation.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def main():
    input_path = pathlib.Path("../dataset/original/exodus")
    output_path = pathlib.Path("../dataset/deskewed/exodus")

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found. Please put dataset in {input_path.absolute()}")

    output_path.mkdir(parents=True, exist_ok=True)

    deskew_images(input_path, output_path)
    crop_images(output_path, output_path)


if __name__ == '__main__':
    main()
