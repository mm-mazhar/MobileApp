import os
import shutil
import time

import cv2
import streamlit as st
from PIL import Image


def cleanup_directory(directory_path):
    """Safely removes a directory and all of its contents."""
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            # Recreate the empty directory
            os.makedirs(directory_path, exist_ok=True)
        except OSError as e:
            st.error(f"Error cleaning up directory {directory_path}: {e.strerror}")


def take_photo(image_dir):
    """
    Captures a photo non-interactively for Streamlit.
    Opens the camera, waits briefly, captures a frame, and saves it.
    """
    filename = os.path.join(image_dir, "captured_image.jpg")

    # Add a spinner for better user experience
    with st.spinner("Activating camera... Please wait."):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open camera. Please check browser/OS permissions.")
            return None

        # Give the camera a moment to initialize and adjust exposure
        time.sleep(1)

        ret, frame = cap.read()

        # Immediately release the camera
        cap.release()
        cv2.destroyAllWindows()
        for i in range(4):
            cv2.waitKey(1)

    if not ret:
        st.error("Failed to capture frame from camera.")
        return None

    # Save the captured frame
    # OpenCV saves in BGR format, but Pillow/st.image will handle it.
    cv2.imwrite(filename, frame)
    st.success(f"Photo captured successfully!")

    return filename


def display_cv2_image(image_path, title=""):
    """
    Displays an image in a Streamlit app.
    Loads an image from a file path and displays it.
    """
    if not os.path.exists(image_path):
        st.error("Image file not found.")
        return

    # st.image can handle file paths directly, or Pillow image objects.
    # Using Pillow is a robust way to ensure format compatibility.
    image = Image.open(image_path)

    st.image(image, caption=title, use_column_width=True)
