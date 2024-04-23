from src.utils.get_feature import move_in_first_quad, convert_to_white_blue
from src.constants.image_constant import BASE_IMAGE_PATH, BASE_STL_FILE_PATH
import os
from stl import mesh
import numpy as np

stl_file_name = 'low_pass_filter.stl'
stl_path = os.path.join(BASE_STL_FILE_PATH, stl_file_name)
image_file_name = os.path.join(BASE_IMAGE_PATH, 'low_pass_filter.png')
#convert_to_white_blue(image_path=image_file_name)
#move_in_first_quad(stl_path, stl_file_name)

stl_data = mesh.Mesh.from_file(stl_path)
triangles = stl_data.vectors
all_vertices = np.concatenate(triangles, axis=0)
max_x = np.max(all_vertices[:, 0])
max_y = np.max(all_vertices[:, 1])
max_z = np.max(all_vertices[:, 2])
print(max_x, max_y, max_z)

min_x = np.min(all_vertices[:, 0])
min_y = np.min(all_vertices[:, 1])
min_z = np.min(all_vertices[:, 2])
print(min_x, min_y, min_z)