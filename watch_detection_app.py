import tensorflow as tf
import zipfile
import os
import gdown
import streamlit as st
import numpy as np
from PIL import Image

def download_and_extract_model():
    model_id = '1EAqFoUexroNtxyrxNXB_eoWYPTRPJw8c'
    url = f'https://drive.google.com/uc?id={model_id}'
    
    zip_path = 'unet-non-aug.zip'
    model_path = 'unet-non-aug.keras'
    
    if not os.path.exists(zip_path):
        st.write("Downloading model zip file...")
        gdown.download(url, zip_path, quiet=False)
    
    if not os.path.exists(model_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
            st.write(f"Model extracted to: {model_path}")
    
    if not os.path.exists(model_path):
        st.error(f"Model extraction failed! Could not find '{model_path}'")
        return None
    
    st.write(f"Model file found at: {model_path}")
    return model_path

def load_model():
    model_path = download_and_extract_model()
    if model_path is None:
        return None
    
    try:
        # Try loading the model with TensorFlow
        model = tf.keras.models.load_model(model_path)
        st.write("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def preprocess_image(image):
    image = image.resize((512, 512))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def segment_image(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    prediction = np.squeeze(prediction, axis=0)
    mask = prediction > 0.5
    mask = np.expand_dims(mask, axis=-1)
    
    gray_image = np.array(image.convert('L'))
    gray_image = np.stack([gray_image] * 3, axis=-1)
    
    result = np.where(mask == 1, np.array(image), gray_image)
    
    return result

def show_image(image):
    st.image(image, channels="RGB", use_column_width=True)

def main():
    st.title("Watch Detection App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        model = load_model()
        if model is None:
            st.error("Failed to load the model. Please check the model file.")
            return

        image = Image.open(uploaded_file)
        segmented_image = segment_image(image, model)

        st.write("Segmented Image:")
        show_image(segmented_image)

if __name__ == "__main__":
    main()
