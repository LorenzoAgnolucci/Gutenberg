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
            mask_even_page = [slice(100, 1205), slice(120, 900)]
            mask_odd_page = [slice(95, 1195), slice(180, 970)]

            crop_img = img[mask_even_page] if (int(image[-5]) % 2 == 0) else img[mask_odd_page]
            cv2.imwrite(os.path.join(output_path, f"{image[:-4]}.png"), crop_img)


