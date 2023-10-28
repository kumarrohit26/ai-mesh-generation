import os
from src.utils.divide_image import divide_imagefile
from src.utils.get_feature import prepare_images_to_generate_test_data
from src.constants.image_constant import *

prepare_images_to_generate_test_data()

for image in os.listdir(ROTATED_IMAGE_PATH):
    image_file_name, _ = os.path.splitext(image)
    stl_file_name = f"{image_file_name}.stl"
    image_file_path = os.path.join(ROTATED_IMAGE_PATH, image)
    stl_file_path = os.path.join(ROTATED_STL_PATH, stl_file_name)
    divide_imagefile(image_file_path, stl_file_path, TILE_SIZE, image_file_name)
    print(f"{image_file_name} Processed...")
