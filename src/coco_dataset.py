import json
import os
import cv2

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
]


def get_annotations(image):
	# TODO: Implement function to get annotations (i.e. stop words bounding boxes and coordinates) for each image
	coco_id_annotation = 1
	return []


def main():
	image_path = "../dataset/deskewed/genesis"
	output_path = "../dataset/stop_words_coco_dataset.json"

	coco_images_output = []
	coco_id_img = 1

	coco_annotations_output = []

	for image_file in sorted(os.listdir(image_path))[1:20]:
		image_input_path = os.path.join(image_path, image_file)
		image = cv2.imread(image_input_path)
		coco_images_output.append({
			"width": image.shape[1],
			"date_captured": "NA",
			"license": 1,
			"flickr_url": "NA",
			"file_name": image_file,
			"id": coco_id_img,
			"coco_url": "",
			"height": image.shape[0]
		})
		coco_id_img += 1

		coco_annotations = get_annotations(image)
		coco_annotations_output.append(coco_annotations)

	coco_output = {
		"info": INFO,
		"licenses": LICENSES,
		"images": coco_images_output,
		"annotations": coco_annotations_output,
		"categories": CATEGORIES
	}

	with open(output_path, "w") as f:
		json.dump(coco_output, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
	main()
