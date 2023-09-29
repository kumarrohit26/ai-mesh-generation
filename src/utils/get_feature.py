import skimage.measure
from skimage import io, color, filters, morphology, measure, util
import numpy as np
from PIL import Image
import os
import open3d as o3d
import shutil
import cv2

def calculate_edges(image_path, tile_size):
        
    # Load the binary image (black for elements, white for background)
    image = io.imread(image_path, as_gray=True)   

    # Calculate the perimeter of the objects in the image
    perimeter = skimage.measure.perimeter(image)

    return perimeter

def calculate_edges_inv(image_path):

    # Load the binary image (replace 'your_image.png' with your image file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Invert the binary image
    inverted_image = cv2.bitwise_not(image)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure that at least one contour was found
    if len(contours) > 0:
        # Get the largest contour (assuming it corresponds to the object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the perimeter of the largest contour
        perimeter = cv2.arcLength(largest_contour, closed=True)
        
    else:
        perimeter = 0
    
    return perimeter

def calculate_disjoint_image(image_path):
    # Load the binary image (black for objects, white for background)
    image = io.imread(image_path, as_gray=True)
    inverted_image = util.invert(image)

    # Apply thresholding to create a binary image
    threshold = filters.threshold_otsu(inverted_image)
    binary_image = inverted_image > threshold

    dilated_image = morphology.binary_dilation(binary_image)
    
    eroded_image = morphology.binary_erosion(dilated_image)
    
    # Label connected components in the binary image
    labeled_image = measure.label(eroded_image)

    # Filter out small regions (adjust the area threshold as needed)
    area_threshold = 100  # Adjust as needed
    filtered_labels = [label for label in range(1, labeled_image.max() + 1) if np.sum(labeled_image == label) >= area_threshold]

    return len(filtered_labels)

def convert_to_white_blue(image_path):
    image = Image.open(image_path)

    # Convert the image to RGB mode (if it's not already)
    image = image.convert('RGB')

    # Get the image dimensions
    width, height = image.size

    # Define the blue color (you can use other color representations as well)
    blue_color = (0, 0, 255)  # (R, G, B)

    # Iterate through each pixel in the image
    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            
            # Check if the pixel is not white
            if pixel != (255, 255, 255):
                # Set the pixel color to blue
                image.putpixel((x, y), blue_color)

    # Save the modified image
    image.save(image_path)

def has_black_pixel(image):
    
    # Convert the image to RGB mode (if it's not already)
    image = image.convert('RGB')

    # Get the image dimensions
    width, height = image.size

    # Initialize a flag to check for black color
    has_black_color = False

    # Iterate through each pixel in the image
    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            
            # Check if the pixel is black (RGB values are 0, 0, 0)
            if pixel == (0, 0, 0):
                has_black_color = True
                break  # No need to continue checking once black color is found
        if has_black_color:
            break  # No need to continue checking once black color is found

    return has_black_color

def rotate_image_and_stl(image_path, stl_path, image_filename, stl_filename, angles):
    rotated_image_path = 'data/rotated_image'
    rotated_stl_path = 'data/rotated_stl'

    for angle in angles:
        # Load the STL file
        mesh = o3d.io.read_triangle_mesh(stl_path)

        # Compute normals for the mesh
        mesh.compute_triangle_normals()

        # Define the axis of rotation
        axis_of_rotation = np.array([0.0, 0.0, 1.0])  # Rotate around the Z-axis
        
        # Convert the angle to radians
        angle_radians = np.radians(angle)

        # Define the rotation matrix
        rot_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            axis_of_rotation
        ], dtype=np.float64)

        # Get mesh vertices as a NumPy array
        vertices = np.asarray(mesh.vertices)

        # Apply the rotation matrix to each vertex
        rotated_vertices = np.dot(vertices, rot_matrix.T)

        # Compute the translation required to move the mesh back into the 1st quadrant
        min_x = np.min(rotated_vertices[:, 0])
        min_y = np.min(rotated_vertices[:, 1])
        translation_vector = [-min_x, -min_y, 0]

        # Translate the rotated vertices
        translated_vertices = rotated_vertices + translation_vector

        # Update the mesh vertices with the rotated vertices
        mesh.vertices = o3d.utility.Vector3dVector(translated_vertices)

        # Save the rotated mesh to a new STL file
        o3d.io.write_triangle_mesh(f'{rotated_stl_path}/{stl_filename}_{angle}.stl', mesh)

        # Open the JPG image file
        image = Image.open(image_path)

        # Rotate the image by 45 degrees
        rotated_image = image.rotate(angle, expand=True)

        # Save the rotated image to a new file
        rotated_image.save(f'{rotated_image_path}/{image_filename}_{angle}.png')

        # Close the original and rotated images
        image.close()
        rotated_image.close()

def copy_file(source_file, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    # Copy the file to the destination folder
    shutil.copy(source_file, destination_folder)

def prepare_images_to_generate_test_data():
    base_image_path = 'data\\images'
    base_stl_file_path = 'data\\stl'
    angles = [15, 30, 45, 60, 75, 90]
    rotated_image_path = 'data\\rotated_image'
    rotated_stl_path = 'data\\rotated_stl'

    for image in os.listdir(base_image_path):
        
        file_name, _ = os.path.splitext(image)
        stl_file_name = f"{file_name}.stl"
        image_path = os.path.join(base_image_path, image)
        stl_path = os.path.join(base_stl_file_path, stl_file_name)
        
        convert_to_white_blue(image_path)
        copy_file(image_path, os.path.join(rotated_image_path))
        copy_file(stl_path, os.path.join(rotated_stl_path))
        rotate_image_and_stl(image_path, stl_path, file_name, file_name, angles)

