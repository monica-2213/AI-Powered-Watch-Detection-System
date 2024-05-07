import streamlit as st
import torch
from PIL import Image
from pathlib import Path

# Load YOLOv8 model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path)
    return model

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
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Watches"):
            # Load YOLOv8 model
            model_path = "https://colab.research.google.com/drive/1eUkh3x9cIm1Gwyv1BQ1v30tk3fwYtsXU?usp=sharing"  # Update this with the path to your exported YOLOv5 model
            model = load_model(model_path)

            # Perform inference
            image = Image.open(uploaded_file)
            results = model(image)

            # Display results
            st.image(results.render(), caption="Detection Results", use_column_width=True)

def about_page():
    st.header("About")
    st.write("""
        This is an AI-powered system designed to detect watches in images.
        It leverages machine learning techniques for object detection to identify watches.
        The system aims to assist collectors, retailers, and enthusiasts in cataloging and appraising watches efficiently.
    """)

if __name__ == "__main__":
    main()
