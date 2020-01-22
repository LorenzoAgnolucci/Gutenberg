import os
import cv2

def crop_images(self, input_path, output_path):
    for image in os.listdir(input_path):
        if image.endswith('.jpg') or image.endswith('png'):
            img = cv2.imread(input_path + '{img_name}'.format(img_name=image))
            # Change if the format of the page is different
            crop_img = img[100:1205, 120:900] if (int(image[:3]) % 2 == 0) else img[95:1195, 180:970]
            cv2.imwrite(output_path + '{img_name}.png'.format(img_name=image[:-4]), crop_img)

