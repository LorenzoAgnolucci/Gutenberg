import os
import pathlib

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
    """
    This functions removes noise (e.g. red embellishments in blue caputs) from single-channel images through
    morphological transformations. To do so two passes of morphological closure are executed, first with a
    vertical structuring element to remove vertical embellishments, then with a standard square structuring element
    to make separated connected components of the same caput come together
    :param channel: single-channel image where the only non-caput coloured pixels are noise
    :return: input channel with most noise removed and where bounding boxes of the caputs' pixels are kept the same
    """
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

        if area >= 200:
            caput_connected_components.append((top, left, bottom, right, area))

    return caput_connected_components


def identify_caput_start(input_path, output_path, max_caput):
    """
    Key assumption when finding caputs: capitals are always left aligned, while caputs are always at the end of a line.
    :param input_path:
    :param output_path:
    :param max_caput: maximum number of caputs to be considered
    :return:
    """
    def is_caput_left_column_end_aligned(left, right):
        """
        :param left: left x-coordinate of caput bounding box
        :param right: right x-coordinate of caput bounding box
        :return: whether the caput is aligned at the right end of the left column
        """
        return 150 < left < 350 or 300 < right < 350

    def is_caput_right_column_end_aligned(left, right):
        """
        :param left: left x-coordinate of caput bounding box
        :param right: right x-coordinate of caput bounding box
        :return: whether the caput is aligned at the right end of the left column
        """
        return 560 < left or 660 < right

    def is_component_near_caput(capital_top, component_top):
        return component_top - 70 < capital_top < component_top + 70

    caput_index = 1

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

        # left paragraph
        for component in red_connected_components:
            top, left, bottom, right, _ = component

            if is_caput_left_column_end_aligned(left, right) and caput_index <= max_caput:  # check whether the connected compontents is alligned to the right of the paragraph (it means it's not a capital letter)
                for capital_component in red_connected_components + blue_connected_components:
                    cap_top, cap_left, cap_bottom, cap_right, _ = capital_component
                    if is_component_near_caput(cap_top, top) and cap_left < 100:  # check if the connected component is not too distance from the capital letter
                        cv2.putText(original_image, f"caput {caput_index}", (right + 5, bottom + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 97, 140), 2)
                        caput_index += 1
                        break

        # right paragraph
        for component in red_connected_components:
            top, left, bottom, right, _ = component

            if is_caput_right_column_end_aligned(left, right) and caput_index <= max_caput:
                for capital_component in red_connected_components + blue_connected_components:
                    cap_top, cap_left, cap_bottom, cap_right, _ = capital_component
                    if is_component_near_caput(cap_top, top) and 300 < cap_left < 510:
                        cv2.putText(original_image, f"caput {caput_index}", (left, bottom + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 97, 140), 2)
                        caput_index += 1
                        break

        cv2.imwrite(os.path.join(output_path, f"{file_name}"), original_image)
        # cv2.imshow(f"{file_name} with CCs", original_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def main():
    input_path = pathlib.Path("../dataset/deskewed/exodus")
    output_path = pathlib.Path("../dataset/caputs/exodus")

    if not input_path.exists():
        raise FileNotFoundError("Deskewed dataset not found. Please run `preprocessing.py` first")

    output_path.mkdir(parents=True, exist_ok=True)

    max_caput = 40

    identify_caput_start(input_path, output_path, max_caput)


if __name__ == '__main__':
    main()
