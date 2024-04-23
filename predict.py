import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load your pre-trained model
model_path = './output/model.h5'
loaded_model = load_model(model_path)

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Preprocess the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((64, 64))  # Resize to the required size
    image = img_to_array(image)
    image = image / 255.0  # Normalize pixel values

    # Expand dimensions to match the model's expected shape
    image = np.expand_dims(image, axis=0)

    # Get predictions
    predictions = loaded_model.predict(image)

    # Assuming you have three classes (adjust accordingly)
    predicted_class = np.argmax(predictions, axis=1)

    # Process the prediction result as needed
    st.write(f"Predicted class: {predicted_class}")