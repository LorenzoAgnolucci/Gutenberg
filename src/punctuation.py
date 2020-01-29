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

    return periods_in_line


def find_middle_periods_in_line(page_image, line_left, line_top, line_right, line_bottom):
    MIDDLE_PERIOD_AREA_LOWER_BOUND = 7
    MIDDLE_PERIOD_AREA_UPPER_BOUND = 35
    MIDDLE_PERIOD_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 13
    MIDDLE_PERIOD_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 18
    MIDDLE_PERIOD_TOP_BOUNDINGBOX_LOWER_BOUND = 10
    BLACK_PIXELS_MIDDLE_PERIOD_HISTOGRAM_THRESHOLD = 8

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
        middle_period_columns_histogram = cv2.reduce(line_image[:, left:right], 0, cv2.REDUCE_SUM,
                                                     dtype=cv2.CV_32F).reshape(
            -1) / 255
        middle_period_black_pixels = stats[label, cv2.CC_STAT_AREA]

        columns_black_pixels_sum = sum(middle_period_columns_histogram) - middle_period_black_pixels

        if MIDDLE_PERIOD_AREA_LOWER_BOUND < area < MIDDLE_PERIOD_AREA_UPPER_BOUND:
            if MIDDLE_PERIOD_BOTTOM_BOUNDINGBOX_LOWER_BOUND < bottom < MIDDLE_PERIOD_BOTTOM_BOUNDINGBOX_UPPER_BOUND \
                    and top > MIDDLE_PERIOD_TOP_BOUNDINGBOX_LOWER_BOUND \
                    and columns_black_pixels_sum < BLACK_PIXELS_MIDDLE_PERIOD_HISTOGRAM_THRESHOLD:
                middle_periods_in_line.append((line_left + left, line_top + top, line_left + right, line_top + bottom))

    return middle_periods_in_line


def find_colons_in_line(page_image, line_left, line_top, line_right, line_bottom):
    COLON_AREA_LOWER_BOUND = 7
    COLON_AREA_UPPER_BOUND = 35
    COLON_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 18
    COLON_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 23
    COLON_TOP_BOUNDINGBOX_LOWER_BOUND = 14
    BLACK_PIXELS_COLON_HISTOGRAM_THRESHOLD = 8
    COLON_OFFSET_LEFT_MARGIN = 3
    COLON_OFFSET_RIGHT_MARGIN = 3
    COLON_OFFSET_AREA_MARGIN = 10
    COLON_UPPER_TOP_THRESHOLD = 3

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
        colon_columns_histogram = cv2.reduce(line_image[:, left:right], 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).reshape(
            -1) / 255
        colon_black_pixels = stats[label, cv2.CC_STAT_AREA]

        columns_black_pixels_sum = sum(colon_columns_histogram) - colon_black_pixels

        if COLON_AREA_LOWER_BOUND < area < COLON_AREA_UPPER_BOUND:
            if COLON_BOTTOM_BOUNDINGBOX_LOWER_BOUND < bottom < COLON_BOTTOM_BOUNDINGBOX_UPPER_BOUND \
                    and top > COLON_TOP_BOUNDINGBOX_LOWER_BOUND \
                    and columns_black_pixels_sum > BLACK_PIXELS_COLON_HISTOGRAM_THRESHOLD:

                for upper_label in range(1, num_components):
                    upper_top = stats[upper_label, cv2.CC_STAT_TOP]
                    upper_bottom = upper_top + stats[upper_label, cv2.CC_STAT_HEIGHT]
                    upper_left = stats[upper_label, cv2.CC_STAT_LEFT]
                    upper_right = upper_left + stats[upper_label, cv2.CC_STAT_WIDTH]

                    upper_area = (upper_bottom - upper_top) * (upper_right - upper_left)

                    if label != upper_label and left - COLON_OFFSET_LEFT_MARGIN < upper_left < left + COLON_OFFSET_LEFT_MARGIN and right - COLON_OFFSET_RIGHT_MARGIN < upper_right < right + COLON_OFFSET_RIGHT_MARGIN \
                            and area - COLON_OFFSET_AREA_MARGIN < upper_area < area + COLON_OFFSET_AREA_MARGIN and upper_top > COLON_UPPER_TOP_THRESHOLD:
                        x, y, width, height = cv2.boundingRect(line_image[:, left:right])

                        colons_in_line.append(
                            (line_left + left, line_top + y, line_left + left + width, line_top + y + height))

    return colons_in_line


def find_long_accents_in_line(page_image, line_left, line_top, line_right, line_bottom):
    LONG_ACCENT_AREA_LOWER_BOUND = 10
    LONG_ACCENT_AREA_UPPER_BOUND = 50
    LONG_ACCENT_BOTTOM_BOUNDINGBOX_LOWER_BOUND = 0
    LONG_ACCENT_BOTTOM_BOUNDINGBOX_UPPER_BOUND = 29
    LONG_ACCENT_TOP_BOUNDINGBOX_LOWER_BOUND = -1
    LONG_ACCENT_TOP_BOUNDINGBOX_UPPER_BOUND = 7
    BLACK_PIXELS_LONG_ACCENT_HISTOGRAM_THRESHOLD = 15
    LONG_ACCENT_WIDTH_BOUNDINGBOX_LOWER_BOUND = 6
    LONG_ACCENT_WIDTH_BOUNDINGBOX_UPPER_BOUND = 15
    LONG_ACCENT_DENSITY_THRESHOLD = 0.0

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
        long_accent_columns_histogram = cv2.reduce(line_image[:, left:right], 0, cv2.REDUCE_SUM,
                                                   dtype=cv2.CV_32F).reshape(
            -1) / 255
        long_accent_black_pixels = stats[label, cv2.CC_STAT_AREA]

        columns_black_pixels_sum = sum(long_accent_columns_histogram) - long_accent_black_pixels
        accent_bounding_box_density = long_accent_black_pixels / area

        if LONG_ACCENT_AREA_LOWER_BOUND < area < LONG_ACCENT_AREA_UPPER_BOUND:
            if LONG_ACCENT_BOTTOM_BOUNDINGBOX_LOWER_BOUND < bottom < LONG_ACCENT_BOTTOM_BOUNDINGBOX_UPPER_BOUND \
                    and LONG_ACCENT_TOP_BOUNDINGBOX_LOWER_BOUND < top < LONG_ACCENT_TOP_BOUNDINGBOX_UPPER_BOUND \
                    and LONG_ACCENT_WIDTH_BOUNDINGBOX_LOWER_BOUND < right - left < LONG_ACCENT_WIDTH_BOUNDINGBOX_UPPER_BOUND \
                    and columns_black_pixels_sum > BLACK_PIXELS_LONG_ACCENT_HISTOGRAM_THRESHOLD \
                    and accent_bounding_box_density > LONG_ACCENT_DENSITY_THRESHOLD:
                long_accents_in_line.append((line_left + left, line_top + top, line_left + right, line_top + bottom))

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
    image_path = "../dataset/deskewed/genesis"
    output_path = pathlib.Path("../dataset/punctuation/genesis/")

    for image in sorted(os.listdir(image_path)):
        draw_punctuation(os.path.join(image_path, image), output_path)

    draw_punctuation(image_path, output_path)


if __name__ == '__main__':
    main()
