import operator
import re
import sys
from math import ceil
from typing import TextIO, List, Tuple

from dtaidistance import dtw
from lines import draw_lines
from punctuation import *

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
    '.': 3,
    ':': 3
}

ALTERNATIVES_LENGTH_MAPPING = {
    "et": 10,
    "quod": 15,
}


def delete_punctuation(line_image):
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


def calimero_pro_edition(bin_image, size_threshold=25):
    """
    Deletes black connected components lower than `size_threshold` on a binarized image
    :param bin_image: binarized input image
    :param size_threshold: connected components with area (no. of black pixels) strictly lower than this threshold are deleted
    :return: binarized image with CC's with area lower than `size_threshold` deleted
    """
    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(bin_image, connectivity=8)
    sizes = stats[:, cv2.CC_STAT_AREA]
    for i in range(1, num_components):
        if sizes[i] < size_threshold:
            bin_image[labels == i] = 0

    return bin_image


def segment_words_in_page(image_path: str, transcription_file: TextIO) -> Tuple[List[Tuple[int, int]], List[List[int]], List[List[List[int]]]]:
    """
    Segments words in a page by doing some preprocessing (see: `calimero_pro_edition` and `delete_punctuation`) and then
    calling `segment_words`
    :param image_path: Path to page image
    :param transcription_file: File with row-by-row transcription
    :return: a triplet `(column_indicators, rows_indicators, page_runs)` where:
     * `column_indicators` (list) contains x-axis coordinates of columns (see: `detect_lines`)
     * `row_indicators` (list of lists) contains a list of y-axis lists of coordinates (one list for each column) (see: `detect_lines`)
     * `page_runs` (list) contains a list of per-row x-coordinates of lines where to draw word separators (`page_runs[col][row]` is the actual list of runs)
    """
    def get_regex_for_column(page_number, column_number):
        if column_number == 0:
            return f"_P{page_number}_C0\n(.*)_P{page_number}_C1"
        return f"_P{page_number}_C1\n(.*)_P{page_number + 1}_C0"

    image_data = cv2.imread(image_path)

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
            line_image = image_data[row_top:row_bottom, column_left:column_right]
            line_without_punctuation = delete_punctuation(line_image)
            calimered_binarized_image = calimero_pro_edition(line_without_punctuation)
            runs = segment_words(calimered_binarized_image, line_image, row_text, page_number, column_index, row_index)
            column_runs.append(runs)

        page_runs.append(column_runs)

    return columns_indicators, rows_indicators, page_runs


def expected_word_lengths_for_line(line_text: str) -> List[List[int]]:
    """
    Computes all the possible words lengths for a line transcription (there may be multiple alternatives due to abbreviations,
    e.g. "et" -> "e", "quod" -> "q;", "dominus" -> "dms", etc.)
    :param line_text: (string) Line transcription
    :return: a list of all possible expected word lengths
    """
    words = line_text.lower().strip("\n =").split(" ")
    expected_lengths_line = [[]]

    for word in words:
        if word in ALTERNATIVES_LENGTH_MAPPING:
            for line_length in expected_lengths_line[:]:
                expected_lengths_line.append(line_length + [sum([LETTER_LENGTH_MAPPING[x] - 1 for x in word])])
                line_length.append(ALTERNATIVES_LENGTH_MAPPING[word])
        else:
            for line_length in expected_lengths_line:
                line_length.append(sum([LETTER_LENGTH_MAPPING[x] - 1 for x in word]))

    return expected_lengths_line


def expected_runs_for_line(line_length_combinations: List[List[int]]) -> List[List[int]]:
    """
    For each word length combinations of a line outputs an histogram that has non-zero values only where a word cut can be made.
    :param line_length_combinations: A list of possible word lengths for a line of text
    :return: An "histogram" that contains a "spike" whenever a word cut can be made
    """
    WORD_CUT_SPIKE = 15
    runs_combinations = []
    for line_length in line_length_combinations:
        runs = [WORD_CUT_SPIKE]
        for run_length in line_length:
            runs += [0] * (run_length + 1)
            runs += [WORD_CUT_SPIKE]
            runs += [0]

        runs.pop()
        runs_combinations.append(runs)

    return runs_combinations


def filter_cuts(shifted_observed_runs: List[int], expected_runs: List[int]) -> Tuple[List[int], List[int], int]:
    """
    Applies dynamic time warping to select cuts in the observed pixel sequence of whitespace runs according to the expected runs
    :param shifted_observed_runs: histogram of whitespace runs encoded as follows: position = start of run, value = length of run
    :param expected_runs: histogram of expected word cuts (see: `expected_runs_for_line`)
    :return: A triplet `(cuts, cuts_indices, distance)` where:
    * cuts: list of x-coordinates of word cuts
    * cuts_indices: indices of selected cuts
    * distance: DTW distance between the two input sequences
    """
    path = dtw.warping_path(expected_runs, shifted_observed_runs)
    distance = dtw.distance(expected_runs, shifted_observed_runs)

    runs_indices = [i for i, x in enumerate(shifted_observed_runs) if x > 0]
    runs_indices.insert(0, 0)

    cuts = []
    cuts_indices = []
    for i, j in path:
        if expected_runs[i] > 0:

            cuts.append(j)
            index_found = j in runs_indices
            if not index_found:
                print(f"DTW associated expected peak in {i} to zero value in {j}", file=sys.stderr)
            else:
                cuts_indices.append(runs_indices.index(j))

    return cuts, cuts_indices, distance


def collapse_histogram(histogram):
    new_histogram = []
    start_count = 0
    i = 0
    while histogram[i] == 1:
        start_count += 1
        new_histogram.append(0)
        i += 1

    if new_histogram:
        new_histogram[start_count - 1] = start_count
        new_histogram = new_histogram[start_count - 1:]

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

    while new_histogram[:][-1] == 0:
        del new_histogram[-1]

    return new_histogram, start_count - 1


def compute_shifted_runs(long_accents_coords, observed_runs):
    shifted_observed_runs = observed_runs.copy()
    WORD_DISTANCE_THRESHOLD = 2  # check whether the following run is large enough, i.e. it's not between two letters
    NUM_ZEROS_ADDED = 15
    coordinates = []
    for long_accent in long_accents_coords:
        coordinates.append(long_accent[2])

    global_offset = 0
    for coordinate in coordinates:
        local_offset = 0
        while coordinate + local_offset < len(observed_runs) and observed_runs[coordinate + local_offset] == 0:
            local_offset += 1
        if coordinate + local_offset < len(observed_runs) and observed_runs[coordinate + local_offset] > WORD_DISTANCE_THRESHOLD:
            shifted_observed_runs[coordinate + local_offset + global_offset: coordinate + local_offset + global_offset] \
                = [0] * NUM_ZEROS_ADDED  # insert NUM_ZEROS_ADDED zeros in the sequence as single elements
            global_offset += NUM_ZEROS_ADDED

    return shifted_observed_runs


def modify_runs_with_periods(periods_coords, observed_runs):
    coordinates = []
    for period_bounding_box in periods_coords:
        coordinates.append((period_bounding_box[0] + period_bounding_box[2]) // 2)

    for coordinate in coordinates:
        found = False
        offset = 0
        while 0 < coordinate + offset < len(coordinates) and not found:
            if observed_runs[coordinate + offset] != 0:
                found = True
            else:
                if offset <= 0:
                    offset = abs(offset) + 1  # poteva piovere
                else:
                    offset *= -1

        if 0 < coordinate + offset < len(coordinates):
            observed_runs[coordinate + offset] *= 2

    return observed_runs


def rescale_expected_runs(expected_runs, observed_runs):
    """
    :param expected_runs:
    :param observed_runs:
    :return: expected runs with the same length as observed_runs, where all nonzero entries are shifted according to the
    ratio of their respective lengths
    """
    scale_factor = len(observed_runs) / len(expected_runs)

    spikes_indices = [i for i, x in enumerate(expected_runs) if x != 0]

    scaled_expected_runs = [0] * len(observed_runs)

    for i in spikes_indices:
        scaled_expected_runs[int(i * scale_factor)] = expected_runs[i]

    return scaled_expected_runs


def segment_words(calimered_line_image, line_image, line_text, page, col, row):
    COLUMN_HISTOGRAM_THRESHOLD = 2
    height, width = calimered_line_image.shape[:2]
    line_histogram = cv2.reduce(calimered_line_image, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).reshape(-1) / 255
    line_histogram = [0 if x > COLUMN_HISTOGRAM_THRESHOLD else 1 for x in line_histogram]
    expected_runs_combinations = expected_runs_for_line(expected_word_lengths_for_line(line_text))
    expected_cuts = len(line_text.strip(" \n=").split(" ")) + 2

    observed_runs, shift_offset = collapse_histogram(line_histogram)
    noise_threshold = min(list(sorted([x for x in observed_runs if x != 0], reverse=True))[:expected_cuts])
    observed_runs = [x if x >= noise_threshold else 0 for x in observed_runs]

    long_accents_coords = find_long_accents_in_line(line_image, 0, 0, width, height)

    periods_coords = find_middle_periods_in_line(line_image, 0, 0, width, height) + \
                     find_periods_in_line(line_image, 0, 0, width, height) + \
                     find_colons_in_line(line_image, 0, 0, width, height)

    observed_runs = modify_runs_with_periods(periods_coords, observed_runs)

    cuts_combinations = []
    observed_runs_shifted = compute_shifted_runs(long_accents_coords, observed_runs)

    for i, expected_runs in enumerate(expected_runs_combinations):
        expected_runs_combinations[i] = rescale_expected_runs(expected_runs, observed_runs)
        cuts_combinations.append(filter_cuts(observed_runs_shifted, expected_runs))

    best_cuts, best_cuts_indices, best_distance = min(cuts_combinations, key=operator.itemgetter(2))

    runs_indices = [i for i, x in enumerate(observed_runs) if x > 0]
    runs_indices.insert(0, 0)
    best_cuts = [runs_indices[i] + shift_offset for i in best_cuts_indices]

    return best_cuts


def draw_word_separators_in_page(image_path, output_path, transcription_file):
    image_output_path = os.path.join(output_path, os.path.basename(image_path))

    columns_indicators, rows_indicators, page_runs = segment_words_in_page(image_path, transcription_file)
    draw_lines(columns_indicators, rows_indicators, image_path, output_path)
    image_data = cv2.imread(image_output_path)

    for column_index, (column_left, column_right) in enumerate(columns_indicators):
        rows_separators = rows_indicators[column_index]

        for row_index, (row_top, row_bottom) in enumerate(zip(rows_separators, rows_separators[1:])):
            row_runs = page_runs[column_index][row_index]

            cv2.putText(image_data, f"{row_index}", (column_left - 15, (row_top + row_bottom) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (238, 0, 255), 1, cv2.LINE_AA)
            for i, end in enumerate(row_runs):
                cv2.line(image_data, (column_left + end, row_top), (column_left + end, row_bottom), (0, 0, 255),
                         1)

    cv2.imwrite(image_output_path, image_data)


def main():
    image_path = "../dataset/deskewed/genesis"
    output_path = pathlib.Path("../dataset/segmentation/genesis/")

    with open("../dataset/genesis1-20.txt") as transcription_file:
        for image in sorted(os.listdir(image_path))[1:20]:
            image_input_path = os.path.join(image_path, image)
            draw_word_separators_in_page(image_input_path, output_path, transcription_file)


if __name__ == '__main__':
    main()
