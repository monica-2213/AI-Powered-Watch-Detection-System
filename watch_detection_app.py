import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load the pre-trained object detection model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.saved_model.load("https://colab.research.google.com/drive/1eUkh3x9cIm1Gwyv1BQ1v30tk3fwYtsXU?usp=sharing")
    return model

# Function to perform object detection on the uploaded image
def detect_objects(image, model):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

def main():
    st.title("AI-Powered Watch Detection System")
    st.sidebar.title("Navigation")

    # Sidebar options
    page = st.sidebar.radio("Go to", ("Upload Image", "About"))

    if page == "Upload Image":
        upload_image_page()
    elif page == "About":
        about_page()

def upload_image_page():
    st.header("Upload Image")
    st.write("Upload an image containing watches to detect and classify them.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Watches"):
            model = load_model()
            # Convert image to numpy array
            image_np = np.array(image)
            # Perform object detection
            detections = detect_objects(image_np, model)
            # Visualization logic to draw bounding boxes on the image
            # Display the image with bounding boxes
            st.image(image, caption="Detection Result", use_column_width=True)

def about_page():
    st.header("About")
    st.write("""
        This is an AI-powered system designed to detect watches in images.
        It leverages machine learning techniques for object detection to identify watches.
        The system aims to assist collectors, retailers, and enthusiasts in cataloging and appraising watches efficiently.
    """)

if __name__ == "__main__":
    main()
