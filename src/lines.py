import os
import pathlib

import cv2
import numpy as np


def binarize_image(image_data):
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    return binary_image


def detect_lines(image_path):
    COLUMN_HISTOGRAM_THRESHOLD = 15
    ROW_HISTOGRAM_THRESHOLD = 50
    ROW_PROXIMITY_THRESHOLD = 10
    FIRST_ROW_SEPARATION_OFFSET = 25
    COLUMN_EXTRA_MARGIN = 14
    ROW_MARGIN_OFFSET = 5
    LINES_IN_PAGE = 42

    print(f"detecting lines on image {image_path}")

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    binary_image = binarize_image(image)

    columns_histogram = cv2.reduce(binary_image, 0, cv2.REDUCE_AVG).reshape(-1)
    columns_histogram = np.array([1 if x > COLUMN_HISTOGRAM_THRESHOLD else 0 for x in columns_histogram])

    columns_indicators = []
    for column, _ in enumerate(columns_histogram[:-1]):
        if abs(columns_histogram[column] - columns_histogram[column + 1]) == 1:
            columns_indicators.append(column)

    """
    Add an extra margin of COLUMN_EXTRA_MARGIN pixels to the right of the two biggest columns
    """
    column_widths = {}
    for column_index in range(len(columns_indicators[:-1])):
        column_widths[(column_index, column_index + 1)] = abs(
            columns_indicators[column_index] - columns_indicators[column_index + 1])

    (first_left, first_right), (second_left, second_right) = list(
        sorted(column_widths, key=column_widths.get, reverse=True))[:2]
    columns_indicators[first_right] += COLUMN_EXTRA_MARGIN
    columns_indicators[second_right] += COLUMN_EXTRA_MARGIN

    columns = []
    for (indicator, _) in enumerate(columns_indicators[:-1]):
        columns.append(binary_image[:, columns_indicators[indicator]:columns_indicators[indicator + 1]])

    rows_histograms = np.zeros((1, height))
    for column in columns:
        rows_histogram_column = cv2.reduce(column, 1, cv2.REDUCE_AVG)
        rows_histogram_column = np.array(
            [1 if x > ROW_HISTOGRAM_THRESHOLD else 0 for x in rows_histogram_column]).reshape((1, -1))
        rows_histograms = np.append(rows_histograms, rows_histogram_column, axis=0)
    rows_histograms = np.delete(rows_histograms, [0], axis=0)

    rows_indicators = []
    for row_indicator in rows_histograms:
        row_section_indicators = []
        for row, _ in enumerate(row_indicator[:-1]):
            if (row_indicator[row] - row_indicator[row + 1]) == 1 and (
                    not row_section_indicators or abs(row_section_indicators[-1] - row) > ROW_PROXIMITY_THRESHOLD):
                row_section_indicators.append(row + ROW_MARGIN_OFFSET)


        if row_section_indicators:  # `row_section_indicators` may be empty in the case of white columns
            row_section_indicators = row_section_indicators[-LINES_IN_PAGE:]
            row_section_indicators.insert(0, row_section_indicators[0] - FIRST_ROW_SEPARATION_OFFSET)

        rows_indicators.append(row_section_indicators)

    columns_indicators = [(columns_indicators[first_left], columns_indicators[first_right]),
                          (columns_indicators[second_left], columns_indicators[second_right])]
    rows_indicators = [rows_indicators[first_left], rows_indicators[second_left]]
    return columns_indicators, rows_indicators


def draw_lines(columns_indicators, rows_indicators, image_path, output_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    for column_index, (column_left, column_right) in enumerate(columns_indicators):
        cv2.line(image, (column_left, 0), (column_left, height), (0, 255, 0), 3)
        cv2.line(image, (column_right, 0), (column_right, height), (0, 255, 0), 3)

        for row_indicator in rows_indicators[column_index]:
            cv2.line(image,
                     (column_left, row_indicator),
                     (column_right, row_indicator),
                     (255, 0, 0),
                     1)

    cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), image)


def main():
    input_path = pathlib.Path("../dataset/deskewed/genesis")
    output_path = pathlib.Path("../dataset/lines/genesis")

    if not input_path.exists():
        raise FileNotFoundError("Deskewed dataset not found. Please run `preprocessing.py` first")

    output_path.mkdir(parents=True, exist_ok=True)

    for file_name in sorted(os.listdir(input_path)):
        image_path = os.path.join(input_path, file_name)

        columns_indicators, rows_indicators = detect_lines(image_path)
        draw_lines(columns_indicators, rows_indicators, image_path, output_path)


if __name__ == '__main__':
    main()
