import tensorflow as tf
import requests
import os
import streamlit as st

# Set up Streamlit page
st.set_page_config(page_title="Watch Segmentation with UNet", page_icon="⌚", layout="wide")

# Define the model download link and local path
model_download_url = "https://drive.google.com/uc?export=download&id=1YWxs3feor6QgdaJwRERY2yqcK4_TGCVK"
model_path = "unet.keras"  # Path where the model will be saved

# Download the model if it doesn't already exist
if not os.path.exists(model_path):
    st.write("Downloading UNet model... Please wait.")
    try:
        # Download the model file
        response = requests.get(model_download_url, stream=True)
        response.raise_for_status()  # Ensure the request was successful
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Model downloaded successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download the model: {e}")
        st.stop()

# Load the model with custom objects
try:
    from tensorflow.keras.utils import register_keras_serializable

    # Define custom loss and metric functions
    smooth = 1e-15

    @register_keras_serializable()
    def iou_coef(y_true, y_pred):
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + smooth) / (union + smooth)

    @register_keras_serializable()
    def iou_loss(y_true, y_pred):
        return 1.0 - iou_coef(y_true, y_pred)

    # Load the model with custom objects
    model = tf.keras.models.load_model(model_path, custom_objects={'iou_loss': iou_loss, 'iou_coef': iou_coef})
    st.success("UNet model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
