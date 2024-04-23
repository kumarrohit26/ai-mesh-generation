import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import os

def load_image(image_path, target_size=(100, 100)):
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Resize the image to the target size (e.g., 224x224 pixels)
        image = cv2.resize(image, target_size)

        # Convert the image to the desired color format (e.g., BGR to RGB)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to the range [0, 1]
        image = image.astype("float32") / 255.0

        # You can apply additional preprocessing steps here, such as data augmentation

        return image

    except Exception as e:
        print(f"Error loading image: {e}")
        return None


# Load image data (images should be preprocessed and loaded into 'image_data')
# Load the CSV file that contains image file paths, feature data, and labels
csv_file_path = "output\\training_data\\train.csv" 
data_df = pd.read_csv(csv_file_path)
# Load labels (number of triangles)

image_names = data_df["name"].tolist()  # List of image file
num_features = 2
feature_data = data_df[['len_of_boundry_inv', 'disjoint_image']].values  # Feature data as a NumPy array
labels = data_df["num_triangles"].values  # Labels as a NumPy array

# Load and preprocess images
image_data = []
for image in image_names:
    img = load_image(os.path.join('output\\tiles', image))
    image_data.append(img)

print((image_data[0].shape), (feature_data.shape))

# Combine image and feature data
combined_data = np.concatenate((image_data, feature_data), axis=3)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_data, labels, test_size=0.2, random_state=42)

# Normalize feature data using StandardScaler
scaler = StandardScaler()
X_train[:, -num_features:] = scaler.fit_transform(X_train[:, -num_features:])
X_test[:, -num_features:] = scaler.transform(X_test[:, -num_features:])

# Define the model
model = keras.Sequential([
    # Convolutional layers for image feature extraction
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    
    # Combine image and feature data
    keras.layers.Concatenate(),
    
    # Regression head
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='linear')  # Linear activation for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Make predictions on new data
# new_data = ...  # Prepare new data (images and features)
# new_data[:, -num_features:] = scaler.transform(new_data[:, -num_features:])
# predictions = model.predict(new_data)

# You can now use 'predictions' to obtain the predicted number of triangles for new data.
