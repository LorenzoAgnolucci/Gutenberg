import os
import pathlib

import cv2
import numpy as np

from lines import binarize_image, detect_lines


def find_periods_in_line(page_image, line_left, line_top, line_right, line_bottom):
    PERIOD_AREA_LOWER_BOUND = 7
    PERIOD_AREA_UPPER_BOUND = 35
    PERIOD_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 18
    PERIOD_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 23
    PERIOD_TOP_BOUNDINGBOX_LOWER_BOUND = 14
    BLACK_PIXELS_PERIOD_HISTOGRAM_THRESHOLD = 8

    periods_in_line = []
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

                periods_in_line.append((line_left + left, line_top + top, line_left + right, line_top + bottom))

    # cv2.imwrite(output_path_image, page_image)
    return periods_in_line


def find_middle_periods_in_line(page_image, line_left, line_top, line_right, line_bottom):
    PERIOD_AREA_LOWER_BOUND = 7
    PERIOD_AREA_UPPER_BOUND = 35
    PERIOD_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 13
    PERIOD_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 18
    PERIOD_TOP_BOUNDINGBOX_LOWER_BOUND = 10
    BLACK_PIXELS_PERIOD_HISTOGRAM_THRESHOLD = 8

    middle_periods_in_line = []

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

                middle_periods_in_line.append((line_left + left, line_top + top, line_left + right, line_top + bottom))

    # cv2.imwrite(output_path_image, page_image)
    return middle_periods_in_line


def find_colons_in_line(page_image, line_left, line_top, line_right, line_bottom):
    PERIOD_AREA_LOWER_BOUND = 7
    PERIOD_AREA_UPPER_BOUND = 35
    PERIOD_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 18
    PERIOD_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 23
    PERIOD_TOP_BOUNDINGBOX_LOWER_BOUND = 14
    BLACK_PIXELS_PERIOD_HISTOGRAM_THRESHOLD = 8

    colons_in_line = []

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


                colons_in_line.append((line_left + left, line_top + y, line_left + left + width, line_top + y + height))

    # cv2.imwrite(output_path_image, page_image)
    return colons_in_line


def find_long_accents_in_line(page_image, line_left, line_top, line_right, line_bottom):
    ACCENT_AREA_LOWER_BOUND = 10
    ACCENT_AREA_UPPER_BOUND = 50
    ACCENT_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 0
    ACCENT_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 29
    ACCENT_TOP_BOUNDINGBOX_LOWER_BOUND = -1
    ACCENT_TOP_BOUNDINGBOX_UPPER_BOUND = 7
    BLACK_PIXELS_ACCENT_HISTOGRAM_THRESHOLD = 15
    ACCENT_WIDTH_BOUNDINGBOX_LOWER_BOUND = 6
    ACCENT_WIDTH_BOUNDINGBOX_UPPER_BOUND = 15
    ACCENT_DENSITY_THRESHOLD = 0.0

    long_accents_in_line = []

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
        accent_black_pixels = stats[label, cv2.CC_STAT_AREA]

        columns_black_pixels_sum = sum(period_columns_histogram) - accent_black_pixels
        accent_bounding_box_density = accent_black_pixels / area

        if ACCENT_AREA_LOWER_BOUND < area < ACCENT_AREA_UPPER_BOUND:
            if ACCENT_BOTTOM_BOUNDINGBOX_LOWER_BOUND < bottom < ACCENT_BOTTOM_BOUNDINGBOX_UPPER_BOUND \
                    and ACCENT_TOP_BOUNDINGBOX_LOWER_BOUND < top < ACCENT_TOP_BOUNDINGBOX_UPPER_BOUND \
                    and ACCENT_WIDTH_BOUNDINGBOX_LOWER_BOUND < right - left < ACCENT_WIDTH_BOUNDINGBOX_UPPER_BOUND \
                    and columns_black_pixels_sum > BLACK_PIXELS_ACCENT_HISTOGRAM_THRESHOLD \
                    and accent_bounding_box_density > ACCENT_DENSITY_THRESHOLD:

                long_accents_in_line.append((line_left + left, line_top + top, line_left + right, line_top + bottom))

    # cv2.imwrite(output_path_image, page_image)
    return long_accents_in_line


def find_dots_over_i_in_line(page_image, line_left, line_top, line_right, line_bottom):
    DOT_AREA_LOWER_BOUND = 5
    DOT_AREA_UPPER_BOUND = 35
    DOT_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 0
    DOT_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 10
    DOT_TOP_BOUNDINGBOX_LOWER_BOUND = -1
    DOT_TOP_BOUNDINGBOX_UPPER_BOUND = 6
    BLACK_PIXELS_DOT_HISTOGRAM_THRESHOLD = 15
    DOT_WIDTH_BOUNDINGBOX_LOWER_BOUND = 0
    DOT_WIDTH_BOUNDINGBOX_UPPER_BOUND = 7
    DOT_DENSITY_UPPER_BOUND = 0.71

    dots_over_i_in_line = []

    binary_image = binarize_image(page_image)

    dots_enhancement_kernel = np.ones((1, 2))

    binary_image = cv2.dilate(binary_image, dots_enhancement_kernel, iterations=1)
    binary_image = cv2.erode(binary_image, dots_enhancement_kernel, iterations=1)

    line_image = binary_image[line_top:line_bottom, line_left:line_right]

    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(line_image, connectivity=8)

    for label in range(1, num_components):
        top = stats[label, cv2.CC_STAT_TOP]
        bottom = top + stats[label, cv2.CC_STAT_HEIGHT]
        left = stats[label, cv2.CC_STAT_LEFT]
        right = left + stats[label, cv2.CC_STAT_WIDTH]

        area = (bottom - top) * (right - left)
        dot_columns_histogram = cv2.reduce(line_image[:, left:right], 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).reshape(
            -1) / 255
        dot_black_pixels = stats[label, cv2.CC_STAT_AREA]
        dot_bounding_box_density = dot_black_pixels / area

        columns_black_pixels_sum = sum(dot_columns_histogram) - dot_black_pixels

        if DOT_AREA_LOWER_BOUND < area < DOT_AREA_UPPER_BOUND:
            if DOT_BOTTOM_BOUNDINGBOX_LOWER_BOUND < bottom < DOT_BOTTOM_BOUNDINGBOX_UPPER_BOUND \
                    and DOT_TOP_BOUNDINGBOX_LOWER_BOUND < top < DOT_TOP_BOUNDINGBOX_UPPER_BOUND \
                    and DOT_WIDTH_BOUNDINGBOX_LOWER_BOUND < right - left < DOT_WIDTH_BOUNDINGBOX_UPPER_BOUND \
                    and columns_black_pixels_sum > BLACK_PIXELS_DOT_HISTOGRAM_THRESHOLD \
                    and dot_bounding_box_density < DOT_DENSITY_UPPER_BOUND:

                dots_over_i_in_line.append((line_left + left, line_top + top, line_left + right, line_top + bottom))

    # cv2.imwrite(output_path_image, page_image)
    return dots_over_i_in_line


def draw_punctuation(image_path, output_path):
    page_image = cv2.imread(image_path)

    columns_indicators, rows_indicators = detect_lines(image_path)

    output_path.mkdir(parents=True, exist_ok=True)

    for column_index, (column_left, column_right) in enumerate(columns_indicators):
        rows_separators = rows_indicators[column_index]
        for row_top, row_bottom in zip(rows_separators, rows_separators[1:]):
            dots_over_i_in_line = find_dots_over_i_in_line(page_image, column_left, row_top, column_right, row_bottom)
            long_accents_in_line = find_long_accents_in_line(page_image, column_left, row_top, column_right, row_bottom)
            colons_in_line = find_colons_in_line(page_image, column_left, row_top, column_right, row_bottom)
            periods_in_line = find_periods_in_line(page_image, column_left, row_top, column_right, row_bottom)
            middle_periods_in_line = find_middle_periods_in_line(page_image, column_left, row_top, column_right,
                                                                 row_bottom)

            for component in dots_over_i_in_line:
                cv2.rectangle(page_image,
                              (component[0], component[1]),
                              (component[2], component[3]),
                              (190, 0, 255),
                              1)


            for component in long_accents_in_line:
                cv2.rectangle(page_image,
                              (component[0], component[1]),
                              (component[2], component[3]),
                              (112, 162, 18),
                              1)

            for component in colons_in_line:
                cv2.rectangle(page_image,
                              (component[0], component[1]),
                              (component[2], component[3]),
                              (0, 0, 255),
                              1)

            for component in periods_in_line:
                cv2.rectangle(page_image,
                              (component[0], component[1]),
                              (component[2], component[3]),
                              (0, 255, 0),
                              1)

            for component in middle_periods_in_line:
                cv2.rectangle(page_image,
                              (component[0], component[1]),
                              (component[2], component[3]),
                              (255, 0, 0),
                              1)

    cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), page_image)


def main():
    image_path = "../dataset/deskewed/genesis/037.png"
    output_path = pathlib.Path("../dataset/periods/genesis/")

    draw_punctuation(image_path, output_path)


if __name__ == '__main__':
    main()
