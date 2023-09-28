import skimage.measure
from skimage import io, color, filters, morphology, measure, util
import numpy as np

def calculate_edges(image_path, tile_size):
        
    # Load the binary image (black for elements, white for background)
    image = io.imread(image_path, as_gray=True)   

    # Calculate the perimeter of the objects in the image
    perimeter = skimage.measure.perimeter(image)

    #pixel = 1
    #roi = slice(pixel, -pixel), slice(pixel, -pixel)
    #perimeter2 = skimage.measure.perimeter(image[roi])
    #print((image[roi] > 0).sum())
    #print("Perimeter of {tile} with perimeter :{perimeter1}, without perimeter {perimeter2}".format(tile=pic, perimeter1=perimeter1, perimeter2=perimeter2))

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