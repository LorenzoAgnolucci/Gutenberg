import json
import os
import pathlib
import re

import cv2

from lines import detect_lines, binarize_image
from words import delete_punctuation, segment_words, calimero_pro_edition, get_regex_for_column, \
    delete_colored_components, segment_words_in_page

INFO = {
    "description": "COCO Dataset Gutenberg Bible",
    "url": "",
    "version": "",
    "year": 2020,
    "contributor": ["Lorenzo Agnolucci", "Alberto Baldrati", "Giovanni Berti"],
    "date_created": "2020-2-10"
}

LICENSES = []

CATEGORIES = [
    {
        "supercategory": "stop word",
        "id": 1,
        "name": "et"
    },
    {
        "supercategory": "stop word",
        "id": 2,
        "name": "in"
    },
    {
        "supercategory": "stop word",
        "id": 3,
        "name": "est"
    },
    {
        "supercategory": "stop word",
        "id": 4,
        "name": "ad"
    },
    {
        "supercategory": "stop word",
        "id": 5,
        "name": "cum"
    },
    {
        "supercategory": "stop word",
        "id": 6,
        "name": "="
    },
    {
        "supercategory": "stop word",
        "id": 7,
        "name": "other"
    }
]


def get_category_id(word):
    category = [cat for cat in CATEGORIES if cat["name"] == word]

    if category:
        return category[0]["id"]
    else:
        return 7  # category of `other word`


def build_word_annotation(points, image_id, category_id, annotation_id):
    top_left, bottom_left, bottom_right, top_right = points
    width = top_right[0] - top_left[0]
    height = bottom_left[1] - top_left[1]
    return {
        "segmentation": [
                top_left[0],
                top_left[1],
                bottom_left[0],
                bottom_left[1],
                bottom_right[0],
                bottom_right[1],
                top_right[0],
                top_right[1]
        ],
        "area": width * height,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": [
            top_left[0],
            top_left[1],
            width,
            height
        ],
        "category_id": category_id,
        "id": annotation_id
    }


def calimerize_line_smart(line_image, line_text):
    line_image_copy = line_image.copy()
    word_cuts = segment_words(line_image_copy, line_text, 0, 0, 0)

    if line_text.strip("\n ")[-1] == "=":
        line_image[:, :word_cuts[-2] + 1] = delete_punctuation(line_image[:, :word_cuts[-2] + 1])
    else:
        line_image[:] = delete_punctuation(line_image)


def get_annotations_in_page(image_path, coco_images_output_path, transcription_file):
    image_data = cv2.imread(image_path)
    raw_image_data = image_data.copy()
    image_data = delete_colored_components(image_data)
    binarized_image_data = binarize_image(image_data)
    page_number = int(os.path.splitext(os.path.basename(image_path))[0])

    columns_indicators, rows_indicators, page_runs = segment_words_in_page(image_path, transcription_file)

    annotations = []
    for column_index, (column_left, column_right) in enumerate(columns_indicators):
        transcription_file.seek(0, 0)
        transcription = transcription_file.read()
        rows_separators = rows_indicators[column_index]

        column_text = re.findall(get_regex_for_column(page_number, column_index), transcription, re.DOTALL)[
            0].splitlines()

        for row_index, (row_top, row_bottom) in enumerate(zip(rows_separators, rows_separators[1:])):
            line_image = binarized_image_data[row_top:row_bottom, column_left:column_right]
            raw_line_image = raw_image_data[row_top:row_bottom, column_left:column_right]
            row_text = column_text[row_index]
            calimerize_line_smart(line_image, row_text)

            word_cuts = page_runs[column_index][row_index]
            words = row_text.strip(" \n").split()

            if "=" in words[-1]:
                words[-1].strip("=")
                words.append("=")

            if len(words) != len(word_cuts) - 1:
                print(f"error page {page_number} column {column_index} row {row_index}")
                raw_line_image[:, :] = (225, 209, 232)
                continue

            for annotation_id, (left, right) in enumerate(zip(word_cuts, word_cuts[1:])):
                x, y, width, height = cv2.boundingRect(line_image[:, left:right])
                top_left = (column_left + left + x, row_top + y)
                bottom_left = (column_left + left + x, row_top + y + height)
                bottom_right = (column_left + left + x + width, row_top + y + height)
                top_right = (column_left + left + x + width, row_top + y)

                points = (top_left, bottom_left, bottom_right, top_right)

                category_id = get_category_id(words[annotation_id])
                annotation = build_word_annotation(points, image_id=page_number, category_id=category_id,
                                                   annotation_id=annotation_id)
                annotations.append(annotation)

    image_output_path = os.path.join(coco_images_output_path, os.path.basename(image_path))
    cv2.imwrite(image_output_path, raw_image_data)

    return annotations


def visualize_annotations(dataset_path, image_path, output_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)
        info = data["info"]
        licenses = data["licenses"]
        categories = data["categories"]
        images = data["images"]
        annotations = data["annotations"]

    for image_data in images:
        image_input_path = os.path.join(image_path, image_data["file_name"])
        image = cv2.imread(image_input_path)
        page_number = int(os.path.splitext(os.path.basename(image_input_path))[0])
        annotation_in_page = [annotation for annotation in annotations if annotation["image_id"] == page_number]

        for annotation in annotation_in_page:
            box = annotation["bbox"]
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 1)

        output_image_path = os.path.join(output_path, str(page_number) + ".jpg")
        cv2.imwrite(output_image_path, image)

def generate_dataset(image_path, output_path, dataset_type, start_page, end_page):
    coco_images_output = []
    coco_annotations_output = []
    with open("../dataset/genesis1-20.txt") as transcription_file:
        for image_file in sorted(os.listdir(image_path))[start_page:end_page]:
            image_input_path = os.path.join(image_path, image_file)

            image = cv2.imread(image_input_path)
            coco_images_output.append({
                "width": image.shape[1],
                "date_captured": "NA",
                "license": 1,
                "flickr_url": "NA",
                "file_name": image_file,
                "id": int(os.path.splitext(os.path.basename(image_input_path))[0]),
                "coco_url": "",
                "height": image.shape[0]
            })

            coco_annotations = get_annotations_in_page(image_input_path, output_path, transcription_file)
            coco_annotations_output += coco_annotations
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "images": coco_images_output,
        "annotations": coco_annotations_output,
        "categories": CATEGORIES
    }
    with open(os.path.join(output_path, f"coco_dataset_{dataset_type}.json"), "w") as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)

def main():
    image_path = "../dataset/deskewed/genesis"
    output_path = pathlib.Path("../dataset/coco_fullpage/")

    output_path.mkdir(parents=True, exist_ok=True)

    generate_dataset(image_path, output_path, "train1-21", start_page=1, end_page=30)
    generate_dataset(image_path, output_path, "validation22-27", start_page=22, end_page=28)
    generate_dataset(image_path, output_path, "test28-33", start_page=28, end_page=34)





if __name__ == '__main__':
    main()
    #image_path = "../dataset/deskewed/genesis"
    #output_path = pathlib.Path("../dataset/coco_fullpage")
    #dataset_path = pathlib.Path("../dataset/coco_fullpage/coco_dataset_train.json")
    #visualize_annotations(dataset_path, image_path, output_path)
