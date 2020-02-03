import operator
import re
import sys
from collections import defaultdict
from math import ceil

from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

from lines import draw_lines
from punctuation import *

# D = 18
# B = 17
LETTER_LENGTH_MAPPING = {
    'a': 11,
    'b': 10,
    'c': 7,
    'd': 9,
    'e': 7,
    'f': 7,
    'g': 13,
    'h': 11,
    'i': 5,
    'l': 5,
    'm': 19,
    'n': 12,
    'o': 11,
    'p': 11,
    'q': 10,
    'r': 8,
    's': 12,
    't': 7,
    'u': 11,
    'v': 10,
    'x': 9,
    'y': 5,
    'z': 11,
    '.': 3
}

ALTERNATIVES_LENGTH_MAPPING = {
    "et": 10,
    "quod": 15,
    'long_s': 9
}


def delete_punctuation(line_image, output_path):
    h, w = line_image.shape[:2]

    bin_image = binarize_image(line_image)

    colons_in_line = find_colons_in_line(line_image, 0, 0, w, h)
    periods_in_line = find_periods_in_line(line_image, 0, 0, w, h)
    middle_periods_in_line = find_middle_periods_in_line(line_image, 0, 0, w, h)

    for component in colons_in_line + periods_in_line + middle_periods_in_line:
        cv2.rectangle(bin_image,
                      (component[0], component[1]),
                      (component[2], component[3]),
                      0,
                      cv2.FILLED)
    return bin_image


def calimero_pro_edition(bin_image, output_path, size_threshold=25):
    # output_path.mkdir(parents=True, exist_ok=True)

    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(bin_image, connectivity=8)
    sizes = stats[:, cv2.CC_STAT_AREA]
    for i in range(1, num_components):
        if sizes[i] < size_threshold:
            bin_image[labels == i] = 0

    # cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), page_image)
    return bin_image


def segment_words_in_page(image_path, output_path, transcription_file):
    def get_regex_for_column(page_number, column_number):
        if column_number == 0:
            return f"_P{page_number}_C0\n(.*)_P{page_number}_C1"
        return f"_P{page_number}_C1\n(.*)_P{page_number + 1}_C0"

    image_data = cv2.imread(image_path)
    # binarized_image = binarize_image(image_data)

    columns_indicators, rows_indicators = detect_lines(image_path)
    page_number = int(os.path.splitext(os.path.basename(image_path))[0])

    page_runs = []
    for column_index, (column_left, column_right) in enumerate(columns_indicators):
        column_runs = []
        transcription_file.seek(0, 0)
        transcription = transcription_file.read()
        rows_separators = rows_indicators[column_index]
        column_text = re.findall(get_regex_for_column(page_number, column_index), transcription, re.DOTALL)[
            0].splitlines()

        for row_index, (row_top, row_bottom) in enumerate(zip(rows_separators, rows_separators[1:])):
            row_text = column_text[row_index]
            line_without_punctuation = delete_punctuation(image_data[row_top:row_bottom, column_left:column_right],
                                                          None)
            calimered_binarized_image = calimero_pro_edition(
                line_without_punctuation, None)
            runs = segment_words(calimered_binarized_image, row_text, page_number, column_index, row_index)
            column_runs.append(runs)

        page_runs.append(column_runs)

    return columns_indicators, rows_indicators, page_runs


def expected_word_lengths_for_line(line_text):
    words = line_text.lower().strip("\n =").split(" ")
    expected_length_words = []

    for word in words:
        expected_length_words.append(sum([LETTER_LENGTH_MAPPING[x] for x in word]))

    return expected_length_words


def expected_runs_for_line(line_lengths):
    runs = [3]
    for run_length in line_lengths:
        runs += [0] * (run_length + 1)
        runs += [3]
        runs += [0]

    return runs


def filter_cuts(observed_runs, expected_runs, page, row, col):
    path = dtw.warping_path(expected_runs, observed_runs)

    cuts = []
    for i, j in path:
        if expected_runs[i] > 0:
            cuts.append(j)

    return cuts


def collapse_histogram(histogram):
    new_histogram = []
    start_count = 0
    i = 0
    while histogram[i] == 1:
        start_count += 1
        new_histogram.append(0)
        i += 1

    if new_histogram:
        new_histogram[0] = start_count

    count = 0
    for index, value in enumerate(histogram[start_count:]):
        if value == 0:
            if count != 0:
                new_histogram[-ceil(count / 2)] = count
                count = 0
            new_histogram.append(0)
        elif value != 0:
            count += 1
            new_histogram.append(0)

    if histogram[-1] == 1:
        new_histogram[-count + 1] = count

    return new_histogram


def segment_words(line_image, line_text, page, col, row):
    line_histogram = cv2.reduce(line_image, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).reshape(-1) / 255
    line_histogram = [0 if x > 0 else 1 for x in line_histogram]
    expected_runs = expected_runs_for_line(expected_word_lengths_for_line(line_text))

    observed_runs = collapse_histogram(line_histogram)

    expected_runs += [0] * (len(observed_runs) - len(expected_runs))

    cuts = filter_cuts(observed_runs, expected_runs, page, row, col)
    return cuts


def draw_word_separators_in_page(image_path, output_path, transcription_file):
    image_output_path = os.path.join(output_path, os.path.basename(image_path))

    columns_indicators, rows_indicators, page_runs = segment_words_in_page(image_path, None, transcription_file)
    draw_lines(columns_indicators, rows_indicators, image_path, output_path)
    image_data = cv2.imread(image_output_path)

    for column_index, (column_left, column_right) in enumerate(columns_indicators):
        rows_separators = rows_indicators[column_index]

        for row_index, (row_top, row_bottom) in enumerate(zip(rows_separators, rows_separators[1:])):
            row_runs = page_runs[column_index][row_index]

            for end in row_runs:
                cv2.line(image_data, (column_left + end, row_top), (column_left + end, row_bottom), (0, 0, 255),
                         1)

    cv2.imwrite(image_output_path, image_data)


def main():
    image_path = "../dataset/deskewed/genesis"
    output_path = pathlib.Path("../dataset/segmentation/genesis/")

    with open("../dataset/genesis1-20.txt") as transcription_file:
        for image in sorted(os.listdir(image_path)):
            image_input_path = os.path.join(image_path, image)
            image_output_path = os.path.join(output_path, image)
            # delete_punctuation(image_input_path, output_path)
            draw_word_separators_in_page(image_input_path, output_path, transcription_file)


if __name__ == '__main__':
    main()
