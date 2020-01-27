import os
import pathlib

import cv2

from lines import binarize_image, detect_lines


def find_periods_in_line(page_image, line_left, line_top, line_right, line_bottom, output_path_image):
    PERIOD_AREA_LOWER_BOUND = 7
    PERIOD_AREA_UPPER_BOUND = 35
    PERIOD_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 18
    PERIOD_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 23
    PERIOD_TOP_BOUNDINGBOX_LOWER_BOUND = 14
    BLACK_PIXELS_PERIOD_HISTOGRAM_THRESHOLD = 8

    binary_image = binarize_image(page_image)
    line_image = binary_image[line_top:line_bottom, line_left:line_right]

    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(line_image, connectivity=8)

    for label in range(1, num_components):
        top = stats[label, cv2.CC_STAT_TOP]
        bottom = top + stats[label, cv2.CC_STAT_HEIGHT]
        left = stats[label, cv2.CC_STAT_LEFT]
        right = left + stats[label, cv2.CC_STAT_WIDTH]

        area = (bottom - top) * (right - left)
        period_columns_histogram = cv2.reduce(line_image[:, left:right], 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).reshape(
            -1) / 255
        period_black_pixels = stats[label, cv2.CC_STAT_AREA]

        columns_black_pixels_sum = sum(period_columns_histogram) - period_black_pixels

        if PERIOD_AREA_LOWER_BOUND < area < PERIOD_AREA_UPPER_BOUND:
            if PERIOD_BOTTOM_BOUNDINGBOX_LOWER_BOUND < bottom < PERIOD_BOTTOM_BOUNDINGBOX_UPPER_BOUND \
                    and top > PERIOD_TOP_BOUNDINGBOX_LOWER_BOUND \
                    and columns_black_pixels_sum < BLACK_PIXELS_PERIOD_HISTOGRAM_THRESHOLD:
                cv2.rectangle(page_image,
                              (line_left + left, line_top + top),
                              (line_left + right, line_top + bottom),
                              (0, 255, 0),
                              1)

    cv2.imwrite(output_path_image, page_image)


def find_middle_periods_in_line(page_image, line_left, line_top, line_right, line_bottom, output_path_image):
    PERIOD_AREA_LOWER_BOUND = 7
    PERIOD_AREA_UPPER_BOUND = 35
    PERIOD_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 13
    PERIOD_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 18
    PERIOD_TOP_BOUNDINGBOX_LOWER_BOUND = 10
    BLACK_PIXELS_PERIOD_HISTOGRAM_THRESHOLD = 8

    binary_image = binarize_image(page_image)
    line_image = binary_image[line_top:line_bottom, line_left:line_right]

    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(line_image, connectivity=8)

    for label in range(1, num_components):
        top = stats[label, cv2.CC_STAT_TOP]
        bottom = top + stats[label, cv2.CC_STAT_HEIGHT]
        left = stats[label, cv2.CC_STAT_LEFT]
        right = left + stats[label, cv2.CC_STAT_WIDTH]

        area = (bottom - top) * (right - left)
        period_columns_histogram = cv2.reduce(line_image[:, left:right], 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).reshape(
            -1) / 255
        period_black_pixels = stats[label, cv2.CC_STAT_AREA]

        columns_black_pixels_sum = sum(period_columns_histogram) - period_black_pixels

        if PERIOD_AREA_LOWER_BOUND < area < PERIOD_AREA_UPPER_BOUND:
            if PERIOD_BOTTOM_BOUNDINGBOX_LOWER_BOUND < bottom < PERIOD_BOTTOM_BOUNDINGBOX_UPPER_BOUND \
                    and top > PERIOD_TOP_BOUNDINGBOX_LOWER_BOUND \
                    and columns_black_pixels_sum < BLACK_PIXELS_PERIOD_HISTOGRAM_THRESHOLD:
                cv2.rectangle(page_image,
                              (line_left + left, line_top + top),
                              (line_left + right, line_top + bottom),
                              (255, 0, 0),
                              1)

    cv2.imwrite(output_path_image, page_image)


def find_colons_in_line(page_image, line_left, line_top, line_right, line_bottom, output_path_image):
    PERIOD_AREA_LOWER_BOUND = 7
    PERIOD_AREA_UPPER_BOUND = 35
    PERIOD_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 18
    PERIOD_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 23
    PERIOD_TOP_BOUNDINGBOX_LOWER_BOUND = 14
    BLACK_PIXELS_PERIOD_HISTOGRAM_THRESHOLD = 8

    binary_image = binarize_image(page_image)
    line_image = binary_image[line_top:line_bottom, line_left:line_right]

    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(line_image, connectivity=8)

    for label in range(1, num_components):
        top = stats[label, cv2.CC_STAT_TOP]
        bottom = top + stats[label, cv2.CC_STAT_HEIGHT]
        left = stats[label, cv2.CC_STAT_LEFT]
        right = left + stats[label, cv2.CC_STAT_WIDTH]

        area = (bottom - top) * (right - left)
        period_columns_histogram = cv2.reduce(line_image[:, left:right], 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).reshape(
            -1) / 255
        period_black_pixels = stats[label, cv2.CC_STAT_AREA]

        columns_black_pixels_sum = sum(period_columns_histogram) - period_black_pixels

        if PERIOD_AREA_LOWER_BOUND < area < PERIOD_AREA_UPPER_BOUND:
            if PERIOD_BOTTOM_BOUNDINGBOX_LOWER_BOUND < bottom < PERIOD_BOTTOM_BOUNDINGBOX_UPPER_BOUND \
                    and top > PERIOD_TOP_BOUNDINGBOX_LOWER_BOUND \
                    and columns_black_pixels_sum > BLACK_PIXELS_PERIOD_HISTOGRAM_THRESHOLD:
                x, y, width, height = cv2.boundingRect(line_image[:, left:right])
                cv2.rectangle(page_image,
                              (line_left + left, line_top + y),
                              (line_left + left + width, line_top + y + height),
                              (0, 0, 255),
                              1)

    cv2.imwrite(output_path_image, page_image)


def find_long_accents_in_line(page_image, line_left, line_top, line_right, line_bottom, output_path_image):
    ACCENT_AREA_LOWER_BOUND = 10
    ACCENT_AREA_UPPER_BOUND = 50
    ACCENT_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 0
    ACCENT_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 29
    ACCENT_TOP_BOUNDINGBOX_LOWER_BOUND = -1
    ACCENT_TOP_BOUNDINGBOX_UPPER_BOUND = 7
    BLACK_PIXELS_ACCENT_HISTOGRAM_THRESHOLD = 15
    ACCENT_WIDTH_BOUNDINGBOX_LOWER_BOUND = 6
    ACCENT_WIDTH_BOUNDINGBOX_UPPER_BOUND = 15


    binary_image = binarize_image(page_image)
    line_image = binary_image[line_top:line_bottom, line_left:line_right]

    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(line_image, connectivity=8)

    for label in range(1, num_components):
        top = stats[label, cv2.CC_STAT_TOP]
        bottom = top + stats[label, cv2.CC_STAT_HEIGHT]
        left = stats[label, cv2.CC_STAT_LEFT]
        right = left + stats[label, cv2.CC_STAT_WIDTH]

        area = (bottom - top) * (right - left)
        period_columns_histogram = cv2.reduce(line_image[:, left:right], 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).reshape(-1) / 255
        accent_black_pixels = stats[label, cv2.CC_STAT_AREA]

        columns_black_pixels_sum = sum(period_columns_histogram) - accent_black_pixels

        if ACCENT_AREA_LOWER_BOUND < area < ACCENT_AREA_UPPER_BOUND:
            if ACCENT_BOTTOM_BOUNDINGBOX_LOWER_BOUND < bottom < ACCENT_BOTTOM_BOUNDINGBOX_UPPER_BOUND \
                    and ACCENT_TOP_BOUNDINGBOX_LOWER_BOUND < top < ACCENT_TOP_BOUNDINGBOX_UPPER_BOUND \
                    and ACCENT_WIDTH_BOUNDINGBOX_LOWER_BOUND < right - left < ACCENT_WIDTH_BOUNDINGBOX_UPPER_BOUND \
                    and columns_black_pixels_sum > BLACK_PIXELS_ACCENT_HISTOGRAM_THRESHOLD:

                cv2.rectangle(page_image,
                              (line_left + left, line_top + top),
                              (line_left + right, line_top + bottom),
                              (255, 0, 0),
                              1)

    cv2.imwrite(output_path_image, page_image)

def find_dots_over_i_in_line(page_image, line_left, line_top, line_right, line_bottom, output_path_image):
    DOT_AREA_LOWER_BOUND = 10
    DOT_AREA_UPPER_BOUND = 30
    DOT_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 0
    DOT_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 29
    DOT_TOP_BOUNDINGBOX_LOWER_BOUND = -1
    DOT_TOP_BOUNDINGBOX_UPPER_BOUND = 7
    BLACK_PIXELS_DOT_HISTOGRAM_THRESHOLD = 15
    DOT_WIDTH_BOUNDINGBOX_LOWER_BOUND = 6
    DOT_WIDTH_BOUNDINGBOX_UPPER_BOUND = 15


    binary_image = binarize_image(page_image)
    line_image = binary_image[line_top:line_bottom, line_left:line_right]

    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(line_image, connectivity=8)

    for label in range(1, num_components):
        top = stats[label, cv2.CC_STAT_TOP]
        bottom = top + stats[label, cv2.CC_STAT_HEIGHT]
        left = stats[label, cv2.CC_STAT_LEFT]
        right = left + stats[label, cv2.CC_STAT_WIDTH]

        area = (bottom - top) * (right - left)
        period_columns_histogram = cv2.reduce(line_image[:, left:right], 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).reshape(-1) / 255
        accent_black_pixels = stats[label, cv2.CC_STAT_AREA]

        columns_black_pixels_sum = sum(period_columns_histogram) - accent_black_pixels

        if DOT_AREA_LOWER_BOUND < area < DOT_AREA_UPPER_BOUND:
            if DOT_BOTTOM_BOUNDINGBOX_LOWER_BOUND < bottom < DOT_BOTTOM_BOUNDINGBOX_UPPER_BOUND \
                    and DOT_TOP_BOUNDINGBOX_LOWER_BOUND < top < DOT_TOP_BOUNDINGBOX_UPPER_BOUND \
                    and DOT_WIDTH_BOUNDINGBOX_LOWER_BOUND < right - left < DOT_WIDTH_BOUNDINGBOX_UPPER_BOUND \
                    and columns_black_pixels_sum > BLACK_PIXELS_DOT_HISTOGRAM_THRESHOLD:

                cv2.rectangle(page_image,
                              (line_left + left-1, line_top + top-1),
                              (line_left + right+1, line_top + bottom+1),
                              (255, 0, 0),
                              1)

    cv2.imwrite(output_path_image, page_image)


def main():
    image_path = "../dataset/deskewed/genesis/025.png"
    output_path = pathlib.Path("../dataset/periods/genesis/")

    image_data = cv2.imread(image_path)

    columns_indicators, rows_indicators = detect_lines(image_path)

    output_path.mkdir(parents=True, exist_ok=True)

    for column_index, (column_left, column_right) in enumerate(columns_indicators):
        rows_separators = rows_indicators[column_index]

        for row_top, row_bottom in zip(rows_separators, rows_separators[1:]):
            find_long_accents_in_line(image_data, column_left, row_top, column_right, row_bottom,
                                      os.path.join(output_path, os.path.basename(image_path)))


if __name__ == '__main__':
    main()
