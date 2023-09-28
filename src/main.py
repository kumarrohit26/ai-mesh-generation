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


print(os.path)
image_path = 'data/images/'
stl_file_path = 'data/stl/'
image_files = ['rotated_image_45']
tile_size = 20

for image in image_files:
    image_file = image_path+image+'.png'
    stl_file = stl_file_path+image+'.stl'
    #print(image_file)
    #print(stl_file)
    DivideImage.divide_imagefile(image_file, stl_file, tile_size, image)
