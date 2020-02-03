import operator
import re

from lines import draw_lines
from punctuation import *


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
    binarized_image = binarize_image(image_data)

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
            line_without_punctuation = delete_punctuation(image_data[row_top:row_bottom, column_left:column_right], None)
            calimered_binarized_image = calimero_pro_edition(
                line_without_punctuation, None)
            runs = segment_words(calimered_binarized_image, row_text)
            column_runs.append(runs)

        page_runs.append(column_runs)

    return columns_indicators, rows_indicators, page_runs


def rlsa(histogram):
    runs = []
    count = 0
    for index, value in enumerate(histogram):
        if value == 0:
            count += 1
        elif value != 0 and count > 0:
            runs.append((index - count, index, count))
            count = 0
    return runs


def segment_words(line_image, line_text):
    words = line_text.strip().split(" ")
    num_words = len(words)
    line_histogram = cv2.reduce(line_image, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).reshape(-1) / 255
    line_histogram = [1 if x > 1 else 0 for x in line_histogram]

    runs = rlsa(line_histogram)

    for run in runs:
        start, end, _ = run
        if start == 0 or end >= len(line_histogram) - 1:
            runs.remove(run)

    candidate_runs = sorted(runs, key=operator.itemgetter(2), reverse=True)[:num_words - 1]
    worst_run_length = candidate_runs[-1]
    ties = [run for run in runs if run[2] == worst_run_length]

    if ties in candidate_runs:
        ties = []

    '''
    if len(ties) > 1:
        while len(candidate_runs) > num_words:
            length_differences = []
            for run in ties:
                run_index = ties.index(run)
                run_before = runs[run_index - 1]
                run_after = runs[run_index + 1]
                fused_word_run_length = run_after[1] - run_before[0] / len(line_histogram)
                word_length = len(words[run_index - 1]) / len(line_text)
                length_difference = abs(fused_word_run_length - word_length)
                length_differences.append(length_difference)

            min_index = length_differences.index(max(length_differences))
            candidate_runs.remove(runs[min_index])
    '''

    return candidate_runs


def draw_word_separators_in_page(image_path, output_path, transcription_file):
    image_data = cv2.imread(image_path)
    image_output_path = os.path.join(output_path, os.path.basename(image_path))

    columns_indicators, rows_indicators, page_runs = segment_words_in_page(image_path, None, transcription_file)
    draw_lines(columns_indicators, rows_indicators, image_path, output_path)
    image_data = cv2.imread(image_output_path)

    for column_index, (column_left, column_right) in enumerate(columns_indicators):
        rows_separators = rows_indicators[column_index]

        for row_index, (row_top, row_bottom) in enumerate(zip(rows_separators, rows_separators[1:])):
            row_runs = page_runs[column_index][row_index]

            for (start, end, count) in row_runs:
                cv2.line(image_data, (column_left + end - 1, row_top), (column_left + end - 1, row_bottom), (0, 0, 255),
                         1)

    cv2.imwrite(image_output_path, image_data)


def main():
    image_path = "../dataset/deskewed/genesis"
    output_path = pathlib.Path("../dataset/segmentation/genesis/")

    with open("../dataset/genesis1-20.txt") as transcription_file:
        for image in sorted(os.listdir(image_path))[1:20]:
            image_input_path = os.path.join(image_path, image)
            image_output_path = os.path.join(output_path, image)
            # delete_punctuation(image_input_path, output_path)
            draw_word_separators_in_page(image_input_path, output_path, transcription_file)


if __name__ == '__main__':
    main()
