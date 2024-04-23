import os, shutil
from src.constants.image_constant import *
dirs = [ROTATED_IMAGE_PATH, ROTATED_STL_PATH, TRAINING_IMAGE_PATH, TRAINING_DATA_FILE_PATH]

for dir in dirs:
    if os.path.isdir(dir):
        shutil.rmtree(dir)
        print('Deleted : ', dir)
