from utils.divide_image import DivideImage
import os
import cv2
import skimage.measure
from skimage import io, color, filters, morphology, measure, util
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
import tripy
import pyvista as pv
from PIL import Image
from utils.get_feature import convert_to_white_blue, prepare_images_to_generate_test_data



image_path = 'data\\rotated_image'
stl_path = 'data\\rotated_stl'

tile_size = 20

#prepare_images_to_generate_test_data()


for image in os.listdir(image_path):
    image_file_name, _ = os.path.splitext(image)
    stl_file_name = f"{image_file_name}.stl"
    image_file_path = os.path.join(image_path, image)
    stl_file_path = os.path.join(stl_path, stl_file_name)
    DivideImage.divide_imagefile(image_file_path, stl_file_path, tile_size, image_file_name)
