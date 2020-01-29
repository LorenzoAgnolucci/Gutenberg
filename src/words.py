import cv2
from punctuation import *


def delete_punctuation(image_path, output_path):
    page_image = cv2.imread(image_path)

    columns_indicators, rows_indicators = detect_lines(image_path)

    output_path.mkdir(parents=True, exist_ok=True)

    for column_index, (column_left, column_right) in enumerate(columns_indicators):
        rows_separators = rows_indicators[column_index]
        for row_top, row_bottom in zip(rows_separators, rows_separators[1:]):
            colons_in_line = find_colons_in_line(page_image, column_left, row_top, column_right, row_bottom)
            periods_in_line = find_periods_in_line(page_image, column_left, row_top, column_right, row_bottom)
            middle_periods_in_line = find_middle_periods_in_line(page_image, column_left, row_top, column_right,
                                                                 row_bottom)

            for component in colons_in_line + periods_in_line + middle_periods_in_line:
                cv2.rectangle(page_image,
                              (component[0], component[1]),
                              (component[2], component[3]),
                              (255, 255, 255),
                              cv2.FILLED)

    cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), page_image)

def main():
    image_path = "../dataset/deskewed/genesis"
    output_path = pathlib.Path("../dataset/segmentation/genesis/")

    for image in sorted(os.listdir(image_path)):
        calimero(os.path.join(image_path, image), output_path)


if __name__ == '__main__':
    main()
