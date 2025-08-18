import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt


def display_cv2_image(image, title=""):
    """
    A helper function to display an OpenCV image (BGR) in a Jupyter Notebook using Matplotlib (RGB).
    """
    # Convert the image from BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))  # Optional: Adjust figure size
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def take_photo(image_dir):
    """
    Opens the default camera, captures a single frame, saves it, and closes.
    """
    filename = os.path.join(image_dir, "captured_image.jpg")

    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Give the camera a moment to initialize and adjust exposure
    print("Camera warming up...")
    time.sleep(1)

    # Capture a single frame
    ret, frame = cap.read()

    # Immediately release the camera
    cap.release()

    if not ret:
        print("Error: Failed to capture frame.")
        cv2.destroyAllWindows()
        return None

    # Destroy any lingering windows
    cv2.destroyAllWindows()
    # Wait for windows to close properly
    for i in range(4):
        cv2.waitKey(1)

    # Save the captured frame
    cv2.imwrite(filename, frame)
    print(f"Image saved to {filename}")

    # Display the captured image in the notebook
    print("\nDisplaying captured image:")
    # Convert from BGR (OpenCV's format) to RGB (Matplotlib's format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_frame)
    plt.axis("off")
    plt.show()

    return filename
