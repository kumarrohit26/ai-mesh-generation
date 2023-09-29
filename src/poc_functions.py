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
import open3d as o3d


def calculate_edges():
    # Load the binary image (black for elements, white for background)
    image = cv2.imread('output/tiles/Microstrip_Coupler_3_5.png', cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    total_length = sum(cv2.arcLength(contour, closed=True) for contour in contours)
    print("Total length of element boundaries:", total_length)



def calculate_edges2():

    #images = ['test_img1.png', 'test_img2.png', 'test_img3.png', 'test_img4.png', 'test_img5.png']
    images = ['Microstrip_Coupler_3_5.png']
    for pic in images:
        # Load the binary image (black for elements, white for background)
        image = io.imread('output/tiles/{pic}'.format(pic=pic), as_gray=True)
        pixel = 1
        roi = slice(pixel, -pixel), slice(pixel, -pixel)
        #print(image[roi].shape)
        # Calculate the perimeter of the objects in the image
        #if (image == 0).sum() == 0:
        #    perimeter1 = 0
        #else:
        #    # Calculate the perimeter of the objects
        #    perimeter1 = skimage.measure.perimeter(image)
        #    perimeter2 = skimage.measure.perimeter(image[roi])
        image_width, image_height = image.shape
        print(image.shape)
        whiteimage = (image_width + image_height - 2)*2
        perimeter1 = skimage.measure.perimeter(image)
        perimeter2 = skimage.measure.perimeter(image[roi])

        print("Perimeter of {tile} with perimeter :{perimeter1}, without perimeter {perimeter2}".format(tile=pic, perimeter1=perimeter1, perimeter2=perimeter2))
    image = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    image = np.ones((9,9), dtype='int')

    image = np.array([
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1]
    ])

    # Calculate the perimeter of the object
    perimeter = skimage.measure.perimeter(image)

    #print("Perimeter of the object:", perimeter)

#calculate_edges2()

def calculate_edges3(image_path):

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
    
    print(f"Perimeter of the object: {perimeter}")

calculate_edges3()

def calculate_disjoint_image():
    # Load the binary image (black for objects, white for background)
    image = io.imread('data/test_images/Microstrip_Coupler_2_3.jpg', as_gray=True)

    # Label connected components in the image
    labeled_image = skimage.measure.label(image)

    # Count the number of disjoint objects
    num_objects = labeled_image.max()

    print("Number of disjoint objects:", num_objects)

    # Print the labeled image for inspection
    #print("Labeled Image:")
    #print(labeled_image)

    # Display the original image and labeled image for visual inspection
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.imshow(image, cmap='gray')
    # plt.title("Original Binary Image")
    # plt.subplot(122)
    # plt.imshow(labeled_image, cmap='nipy_spectral')
    # plt.title("Labeled Image")
    # plt.show()

    #print("Number of disjoint objects:", labeled_image.min())

    # Convert to grayscale
    #grayscale_image = color.rgb2gray(image)
    #plt.imshow(grayscale_image, cmap='nipy_spectral')
    #plt.show()

    # Apply thresholding to create a binary image
    inverted_image = util.invert(image)
    threshold = filters.threshold_otsu(inverted_image)
    binary_image = inverted_image > threshold

    dilated_image = morphology.binary_dilation(binary_image)
    plt.imshow(dilated_image, cmap='nipy_spectral')
    plt.show()
    
    eroded_image = morphology.binary_erosion(dilated_image)
    plt.imshow(eroded_image, cmap='nipy_spectral')
    plt.show()
    
    # Label connected components in the binary image
    labeled_image = measure.label(eroded_image)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(eroded_image)
    plt.title("Original Binary Image")
    plt.subplot(122)
    plt.imshow(labeled_image, cmap='gray')
    plt.title("Labeled Image")
    plt.show()

    # Filter out small regions (adjust the area threshold as needed)
    area_threshold = 100  # Adjust as needed
    filtered_labels = [label for label in range(1, labeled_image.max() + 1) if np.sum(labeled_image == label) >= area_threshold]
    num_black_objects = sum(binary_image[labeled_image == label].any() for label in filtered_labels)

    print("Number of disjoint objects:", len(filtered_labels))
    print("Number of black objects:", (num_black_objects))

#calculate_disjoint_image()


def rotate_stl_image():

    # Load the STL file
    stl_mesh = mesh.Mesh.from_file('data/stl/Microstrip_Coupler.stl')

    # Define the rotation angle (in degrees) and axis of rotation
    angle_degrees = 45 # Adjust this as needed
    axis_of_rotation = [0, 0, 1]  # Rotate around the Z-axis
    point_of_rotation = [90,70, 10]

    # Perform the rotation
    stl_mesh.rotate(axis_of_rotation, angle_degrees, point_of_rotation)

    # Save the rotated mesh in ASCII format
    stl_mesh.save('data/stl/rotated_file_ascii.stl')

    # # Open the JPG image file
    # image = Image.open('data/images/Microstrip_Coupler.png')

    # # Rotate the image by 45 degrees
    # rotated_image = image.rotate(45, expand=True)

    # # Save the rotated image to a new file
    # rotated_image.save('rotated_image.png')

    # # Close the original and rotated images
    # image.close()
    # rotated_image.close()

def rotate_stl_image2():

    # Load the STL file
    mesh = o3d.io.read_triangle_mesh('data/stl/Microstrip_Coupler.stl')

    # Compute normals for the mesh
    mesh.compute_triangle_normals()


    # Define the rotation angle (in degrees) and axis of rotation
    angle_degrees = 45.0 # Adjust this as needed
    axis_of_rotation = np.array([0.0, 0.0, 1.0])  # Rotate around the Z-axis
    point_of_rotation = [90,70, 10]

    
    # Convert the angle to radians
    angle_radians = np.radians(angle_degrees)

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
    o3d.io.write_triangle_mesh(f'data/stl/rotated_file_{angle_degrees}.stl', mesh)

    # Open the JPG image file
    image = Image.open('data/images/Microstrip_Coupler.png')

    # Rotate the image by 45 degrees
    rotated_image = image.rotate(45, expand=True)

    # Save the rotated image to a new file
    rotated_image.save(f'data/images/rotated_image_{angle_degrees}.png')

    # Close the original and rotated images
    image.close()
    rotated_image.close()


def check_image_size():
    image_path = ['data/images/Microstrip_Coupler.png', 'rotated_image.png']
    for image in image_path:
        load_image = Image.open(image)
        rgb_im = load_image.convert('RGB')
        image_width, image_height = rgb_im.size
        print(image_width, image_height)


def rotate_image():
    # Open the JPG image file
    image = Image.open('data/images/Microstrip_Coupler.png')
    angle_of_rtation = 45
    # Rotate the image by 45 degrees
    rotated_image = image.rotate(angle_of_rtation, expand=True)

    # Save the rotated image to a new file
    rotated_image.save(f'data/images/Microstrip_Coupler_{angle_of_rtation}.png')

    # Close the original and rotated images
    image.close()
    rotated_image.close()

#rotate_stl_image()

def convert_to_white_blue():
    image = Image.open('data/images/Microstrip_Coupler.png')

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
    image.save('data/images/Microstrip_Coupler.png')

#convert_to_white_blue()
#rotate_image()

def has_black_pixel():
    # Load the image
    image = Image.open('output/tiles/Microstrip_Coupler_45_1_5.png')

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

    # Check the flag to determine if black color was found
    if has_black_color:
        print("The image contains black color.")
    else:
        print("The image does not contain black color.")

#has_black_pixel()