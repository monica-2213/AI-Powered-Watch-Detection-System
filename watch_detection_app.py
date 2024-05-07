import streamlit as st

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
            # Placeholder for detection logic
            st.write("Watch detection results will be displayed here.")

def about_page():
    st.header("About")
    st.write("""
        This is an AI-powered system designed to detect watches in images.
        It leverages machine learning techniques for object detection to identify watches.
        The system aims to assist collectors, retailers, and enthusiasts in cataloging and appraising watches efficiently.
    """)

if __name__ == "__main__":
    main()
