from PIL import Image
import numpy as np
import os
import pandas as pd
from stl import mesh
from .get_feature import calculate_edges, calculate_disjoint_image, has_black_pixel, calculate_edges_inv, categorize, calculate_neighbours, get_category
import csv
from src.constants.image_constant import TRAINING_DATA_FILE, Z_AXIS_TOLERANCE, TRAINING_IMAGE_PATH, TRAINING_DATA_FILE_PATH, TRAINING_DATA_FILE


def calculate_area_of_triangle(triangle_vertices):
    side1 = triangle_vertices[1] - triangle_vertices[0]
    side2 = triangle_vertices[2] - triangle_vertices[0]
    cross_product = np.cross(side1, side2)
    area = 0.5 * np.linalg.norm(cross_product)
    return area

def resize_image(image, target_height):
    height_percent = target_height / float(image.size[1])
    target_width = int(float(image.size[0]) * float(height_percent))
    resized_image = image.resize((target_width, target_height))
    return resized_image


def divide_imagefile(image_path, stl_file_path, tile_size, image_name, z_axis_data):
    
    # Load the STL file
    stl_data = mesh.Mesh.from_file(stl_file_path)
    triangles = stl_data.vectors

    # Flatten the vertices to a single array
    all_vertices = np.concatenate(triangles, axis=0)

    # Calculate the maximum values for each axis
    # max_x,max_y represents the width and height of image in stl file. taking max as the min is 0
    # made sure that the rotated stl image is always in 1st quadrant touching x and y axis.
    max_x = np.max(all_vertices[:, 0])
    max_y = np.max(all_vertices[:, 1])

    z_top = z_axis_data['z_top']
    z_bottom = z_axis_data['z_bottom']
    height = z_top - z_bottom
    Z_AXIS = z_bottom + (Z_AXIS_TOLERANCE*height)

    z_axis_triangles = []

    for triangle in triangles:
        vertex_1, vertex_2, vertex_3 = np.array_split(triangle, 3)
        vertex_1 = vertex_1.flatten()
        vertex_2 = vertex_2.flatten()
        vertex_3 = vertex_3.flatten()
        if (vertex_1[2] >= Z_AXIS) and (vertex_2[2] >= Z_AXIS) and (vertex_3[2] >= Z_AXIS):
            z_axis_triangles.append(triangle)
        
    z_axis_triangles = np.array(z_axis_triangles)

    image = Image.open(image_path)
    #rgb_im = image.convert('RGB')

    resized_image = resize_image(image, 2000)
    image_width, image_height = resized_image.size
    
    x_ratio = int(image_width / max_x)
    y_ratio = int(image_height / max_y)

    num_cols = image_width // tile_size
    num_rows = image_height // tile_size

    if not os.path.exists(TRAINING_DATA_FILE_PATH):
        os.makedirs(TRAINING_DATA_FILE_PATH)
    
    if not os.path.exists(TRAINING_IMAGE_PATH):
        os.makedirs(TRAINING_IMAGE_PATH)

    csv_file_path = os.path.join(TRAINING_DATA_FILE_PATH, TRAINING_DATA_FILE)
    
    name = list()
    len_of_boundry = list()
    len_of_boundry_inv = list()
    tile_size_list = list()
    disjoint_image = list()
    final_num_triangles = list()
    image_category = list()
    category = list()

    
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * tile_size/x_ratio
            upper = row * tile_size/y_ratio
            right = left + tile_size/x_ratio
            lower = upper + tile_size/y_ratio
            triangle_section_map = {}
            for i in z_axis_triangles:
                for vertex in i:
                    x, y, _ = vertex
                    if (left) <= x < (right) and (upper) <= y < (lower):
                        triangle_section_map.setdefault((row, col), []).append(i)
                        break
            try:
                num_triangles = len(triangle_section_map[(row, col)])
            except KeyError:
                num_triangles = 0

            crop_left = col * tile_size
            crop_right = crop_left + tile_size
            crop_upper = (num_rows-1-row)*(tile_size)
            crop_lower = crop_upper + (tile_size)
            tile = resized_image.crop((crop_left, crop_upper, crop_right, crop_lower))

            if num_triangles > 0 and not has_black_pixel(tile):
                tile = tile.convert('RGB')

                # Get the dimensions of the tile
                width, height = tile.size

                # Iterate through each pixel in the cropped tile and set non-white pixels to black
                for x in range(width):
                    for y in range(height):
                        pixel = tile.getpixel((x, y))
                        if pixel != (255, 255, 255):  # Check if the pixel is not white
                            tile.putpixel((x, y), (0, 0, 0))  # Set non-white pixel to black

                # Save the tile with a unique name
                
                filename = f'{image_name}_{row}_{col}.png'
                
                tile.save(os.path.join(TRAINING_IMAGE_PATH, filename))
                output_image_path = f'{TRAINING_IMAGE_PATH}/{filename}'

                name.append(filename)
                len_of_boundry.append(calculate_edges(image_path = output_image_path))
                len_of_boundry_inv.append(calculate_edges_inv(image_path = output_image_path))
                disjoint_image.append(calculate_disjoint_image(image_path = output_image_path))
                #num_pixels = (tile_size*x_ratio) * (tile_size * y_ratio)
                tile_size_list.append(tile_size*x_ratio)
                image_category.append(categorize(num_triangles))
                final_num_triangles.append(num_triangles)
                category.append(get_category(num_triangles))

    features_df = pd.DataFrame({
                    'name': name,
                    'len_of_boundry': len_of_boundry,
                    'len_of_boundry_inv': len_of_boundry_inv,
                    'disjoint_image': disjoint_image,
                    'tile_size': tile_size_list,
                    'num_triangles': final_num_triangles,
                    'image_category': image_category
                })
    if features_df.shape[0] > 0:
        features_df['neighbours'] = features_df.apply(lambda row: calculate_neighbours(row, num_rows, num_cols, image_name, features_df), axis=1)

        if not os.path.exists(csv_file_path):
            features_df.to_csv(csv_file_path, index=False)
        else:
            features_df.to_csv(csv_file_path, mode='a', header=False, index=False)

