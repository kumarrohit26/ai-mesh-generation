import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.preprocessing import LabelEncoder

# Specify GPU utilization
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Set GPU memory growth to avoid resource allocation issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def categorize(value):
    if 0 <= value <= 100:
        return 0
    elif 101 <= value <= 200:
        return 1
    else:
        return 2

def add_categories():
    file_path = 'output\\training_data\\train.csv'
    df = pd.read_csv(file_path)
    # Apply the categorize function to the numerical column and create a new column
    df['Category'] = df['num_triangles'].apply(categorize)

    # Save the DataFrame with the new column back to a CSV file
    df.to_csv('output_file.csv', index=False)

file_path = "output_file.csv"
data = pd.read_csv(file_path)

# Separate image file names, features, and labels
# ['name','len_of_boundry','len_of_boundry_inv','tile_size', 'disjoint_image','num_triangles']
image_file_names = data['name'].values
features = data[['len_of_boundry_inv','tile_size', 'disjoint_image']].values
labels = data['category'].values

#image_dir = 'output/tiles/'
#data['image_path'] = image_dir + data['name']
#print(data.head(3))
# Convert the "Category" column to string type
#data['Category'] = data['Category'].astype(str)

# Encode categorical labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Load and preprocess black and white images
image_data = []
for image_file in image_file_names:
    # Load, preprocess, and resize the images
    image = tf.keras.preprocessing.image.load_img('output/tiles/' + image_file, target_size=(224, 224), color_mode='grayscale')
    image = tf.keras.preprocessing.image.img_to_array(image)
    image /= 255.0  # Normalize pixel values

    image_data.append(image)

# Create data generators for black and white images
image_data = np.array(image_data)
image_datagen = ImageDataGenerator(validation_split=0.2)
image_train_gen = image_datagen.flow(image_data, labels, batch_size=32, subset='training')
image_val_gen = image_datagen.flow(image_data, labels, batch_size=32, subset='validation')


datagen = ImageDataGenerator(rescale=1./255)

image_generator = datagen.flow_from_dataframe(
    data,
    x_col="image_path",
    y_col="Category",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

num_classes = 3

# Create a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

print(model.summary())

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the generator
model.fit(image_generator, epochs=10)