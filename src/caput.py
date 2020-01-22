import os

import cv2
import numpy as np


def find_caput_pixels(page_image_path):
    img = cv2.imread(page_image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 150, 100])
    upper_red = np.array([15, 255, 255])

    lower_blue = np.array([90, 100, 70])
    upper_blue = np.array([140, 200, 150])

    red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    red_channel = cv2.bitwise_and(img, img, mask=red_mask)
    blue_channel = cv2.bitwise_and(img, img, mask=blue_mask)

    _, red_binarized_channel = cv2.threshold(red_channel, 127, 255, cv2.THRESH_BINARY)
    _, blue_binarized_channel = cv2.threshold(blue_channel, 127, 255, cv2.THRESH_BINARY)

    return red_binarized_channel, blue_binarized_channel


def morphological_denoise(channel):
    kernel = np.ones((3, 3))

    noise_removal_kernel = np.ones((2, 1))

    channel = cv2.erode(channel, noise_removal_kernel, iterations=1)
    channel = cv2.dilate(channel, noise_removal_kernel, iterations=1)
    channel = cv2.dilate(channel, kernel, iterations=20)
    channel = cv2.erode(channel, kernel, iterations=20)

    return channel


def main():
    input_path = "../dataset/deskewed/exodus"

    for file_name in sorted(os.listdir(input_path)):
        red_ch, blue_ch = find_caput_pixels(os.path.join(input_path, file_name))

        red_ch = morphological_denoise(red_ch)
        blue_ch = morphological_denoise(blue_ch)

        cv2.imshow(f"{file_name} blue", blue_ch)
        cv2.waitKey(0)
        cv2.imshow(f"{file_name} red", red_ch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
