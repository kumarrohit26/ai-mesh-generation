from PIL import Image
import numpy as np
import os
from stl import mesh
from .get_feature import calculate_edges, calculate_disjoint_image, has_black_pixel, calculate_edges_inv, categorize
import csv
# import cv2
from src.constants.image_constant import TRAINING_DATA_FILE, Z_AXIS_TOLERANCE, TRAINING_IMAGE_PATH, TRAINING_DATA_FILE_PATH, TRAINING_DATA_FILE
Z_AXIS = 9.9


def calculate_area_of_triangle(triangle_vertices):
    side1 = triangle_vertices[1] - triangle_vertices[0]
    side2 = triangle_vertices[2] - triangle_vertices[0]
    cross_product = np.cross(side1, side2)
    area = 0.5 * np.linalg.norm(cross_product)
    return area


def divide_imagefile(image_path, stl_file_path, tile_size, image_name):
    
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
    max_z = np.max(all_vertices[:, 2])
    max_z = Z_AXIS_TOLERANCE*max_z

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
    rgb_im = image.convert('RGB')
    image_width, image_height = rgb_im.size

    x_ratio = int(image_width / max_x)
    y_ratio = int(image_height / max_y)

    num_cols = image_width // (tile_size * x_ratio)
    num_rows = image_height // (tile_size * y_ratio)

    if not os.path.exists(TRAINING_DATA_FILE_PATH):
        os.makedirs(TRAINING_DATA_FILE_PATH)
    
    if not os.path.exists(TRAINING_IMAGE_PATH):
        os.makedirs(TRAINING_IMAGE_PATH)

    csv_file_path = os.path.join(TRAINING_DATA_FILE_PATH, TRAINING_DATA_FILE)
    file_exists = os.path.isfile(csv_file_path)
    f = open(csv_file_path, 'a')
    header = ['name','len_of_boundry','len_of_boundry_inv','tile_size', 'disjoint_image','num_triangles','image_category']
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(header)

    for row in range(num_rows):
        for col in range(num_cols):
            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size
            triangle_section_map = {}
            for i in z_axis_triangles:
                for vertex in i:
                    x, y, _ = vertex
                    if (left) <= x < (right) and (upper) <= y < (lower):
                        triangle_section_map.setdefault((row, col), []).append(i)
                        break
            try:
                num_triangles = len(triangle_section_map[(row, col)])
                #print('{row},{col} - {length}'.format(row=row, col=col, length=num_triangles), end=' | ')
            except KeyError:
                num_triangles = 0
                #print('{row},{col} - {length}'.format(row=row, col=col, length=num_triangles), end=' | ')
            # Crop the tile from the image
            crop_left = left*x_ratio
            crop_right = right*x_ratio
            crop_upper = (num_rows-1-row)*y_ratio*tile_size
            crop_lower = (num_rows-row)*y_ratio*tile_size
            tile = rgb_im.crop((crop_left, crop_upper, crop_right, crop_lower))

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
                len_of_boundry = calculate_edges(image_path = output_image_path)
                len_of_boundry_inv = calculate_edges_inv(image_path = output_image_path)
                num_of_disjoint_image = calculate_disjoint_image(image_path = output_image_path)
                num_pixels = (tile_size*x_ratio) * (tile_size * y_ratio)
                image_category = categorize(num_triangles)
                data = [filename, len_of_boundry,len_of_boundry_inv, tile_size*x_ratio, num_of_disjoint_image, num_triangles, image_category]
                writer.writerow(data)
    f.close()
