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

    red_channel = cv2.cvtColor(cv2.cvtColor(red_channel, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    blue_channel = cv2.cvtColor(cv2.cvtColor(blue_channel, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

    _, red_binarized_channel = cv2.threshold(red_channel, 127, 255, cv2.THRESH_BINARY)
    _, blue_binarized_channel = cv2.threshold(blue_channel, 30, 255, cv2.THRESH_BINARY)

    return red_binarized_channel, blue_binarized_channel


def morphological_denoise(channel):
    kernel = np.ones((3, 3))

    noise_removal_kernel = np.ones((2, 1))

    channel = cv2.erode(channel, noise_removal_kernel, iterations=1)
    channel = cv2.dilate(channel, noise_removal_kernel, iterations=1)
    channel = cv2.dilate(channel, kernel, iterations=20)
    channel = cv2.erode(channel, kernel, iterations=20)

    return channel


def find_caput_connected_components(denoised_channel):
    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(denoised_channel, connectivity=8)
    caput_connected_components = []

    for label in range(1, num_components):
        top = stats[label, cv2.CC_STAT_TOP]
        bottom = top + stats[label, cv2.CC_STAT_HEIGHT]
        left = stats[label, cv2.CC_STAT_LEFT]
        right = left + stats[label, cv2.CC_STAT_WIDTH]

        area = (bottom - top) * (right - left)

        if area >= 100:
            caput_connected_components.append((top, left, bottom, right, area))

    return caput_connected_components


def main():
    input_path = "../dataset/deskewed/exodus"

    for file_name in sorted(os.listdir(input_path)):
        red_ch, blue_ch = find_caput_pixels(os.path.join(input_path, file_name))

        red_ch = morphological_denoise(red_ch)
        blue_ch = morphological_denoise(blue_ch)

        red_connected_components = find_caput_connected_components(red_ch)
        blue_connected_components = find_caput_connected_components(blue_ch)

        original_image = cv2.imread(os.path.join(input_path, file_name))

        for component in (red_connected_components + blue_connected_components):
            top, left, bottom, right, _ = component
            cv2.rectangle(original_image, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow(f"{file_name} with CCs", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
