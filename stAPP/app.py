import os

import google.generativeai as genai
import matplotlib.pyplot as plt
import requests
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
from PIL import Image
from sympy import content
from utils.st_utils import cleanup_directory, take_photo

print(f"INFO | Loading .env file success: {load_dotenv(override=True)}")
# print(f"Google API Key: {os.getenv('GOOGLE_API_KEY')[-5:]}")

# ======================================================================================
# 0. App Configuration
# ======================================================================================
st.set_page_config(
    page_title="SightGuide Assistant",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("👁️ SightGuide Assistant")
st.markdown(
    "Your AI assistant for understanding your surroundings. Capture a photo, and AI Model will describe the scene and identify any obstacles."
)

# ======================================================================================
# 1. Helper Functions (Adapted from your utils)
# ======================================================================================

# --- Directory and File Management ---
IMAGES_DIR = "./stAPP/images"
SAMPLE_IMAGES = "./stAPP/sample_images"
TTS_OUTPUT_DIR = "./stAPP/tts_output"

# Create directories on startup if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)


# ======================================================================================
# 2. Gemini Configuration and Model Prompt
# ======================================================================================
try:
    # GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    MODEL = genai.GenerativeModel(model_name="models/gemma-3-4b-it")
except Exception:
    st.error(
        "Google API Key not found. Please add it to your Streamlit secrets.", icon="🚨"
    )
    st.stop()

PROMPT = """
You are an assistant for a visually impaired person.
Analyze the attached image from a first-person perspective.
Describe the scene, focusing on any immediate obstacles in the path ahead.
Be clear, concise, and mention the approximate location of the obstacles (e.g., 'directly in front,' 'to your left').
Provide a safe direction to move, if possible.
"""

# ======================================================================================
# 3. App State Management (The Core of the App)
# ======================================================================================

# Initialize session state variables
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "description" not in st.session_state:
    st.session_state.description = ""
if "tts_path" not in st.session_state:
    st.session_state.tts_path = ""


def reset_app_state():
    """Resets the session state and cleans up files."""
    st.session_state.image_path = None
    st.session_state.analysis_complete = False
    st.session_state.description = ""
    st.session_state.tts_path = ""
    cleanup_directory(IMAGES_DIR)
    cleanup_directory(TTS_OUTPUT_DIR)
    st.success("App has been reset.")


# ======================================================================================
# 4. Main App UI and Logic
# ======================================================================================

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    if st.button("Start Over", use_container_width=True):
        reset_app_state()

st.divider()

# --- Step 1: Provide an Image (Upload or Capture) ---
st.subheader("Step 1: Provide an Image")

tab1, tab2 = st.tabs(["🖼️ Upload an Image", "📸 Take a Photo"])

with tab1:
    with st.container(border=True):
        # --- UPLOAD OPTION ---
        st.write("**Upload an image from your device**")
        uploaded_file = st.file_uploader(
            "Supports JPG, JPEG, and PNG formats.",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )

        st.markdown(
            "<h3 style='text-align: center; color: grey;'>OR</h3>",
            unsafe_allow_html=True,
        )

        # --- SAMPLE IMAGE OPTION ---
        st.write("**Use a sample image for testing**")
        st.markdown("No image handy? Try this sample photo of a common obstacle.")

        if st.button("Load Sample Image", use_container_width=True):
            SAMPLE_IMAGE_URL = (
                "https://images.pexels.com/photos/3779198/pexels-photo-3779198.jpeg"
            )
            SAMPLE_IMAGE_FILENAME = "downloaded_image.jpg"
            save_path = os.path.join(IMAGES_DIR, SAMPLE_IMAGE_FILENAME)

            if not os.path.exists(save_path):
                with st.spinner("Downloading sample image..."):
                    try:
                        response = requests.get(SAMPLE_IMAGE_URL)
                        response.raise_for_status()
                        with open(save_path, "wb") as f:
                            f.write(response.content)
                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to download sample image: {e}")
                        st.stop()

            st.session_state.image_path = save_path
            st.session_state.analysis_complete = False
            st.success("Sample image loaded!")
            st.rerun()

    # Logic to handle the user's own uploaded file
    if uploaded_file is not None:
        save_path = os.path.join(IMAGES_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.session_state.image_path != save_path:
            st.session_state.image_path = save_path
            st.session_state.analysis_complete = False
            st.rerun()

with tab2:
    with st.container(border=True):
        st.write("**Capture a live photo from your webcam**")
        st.info(
            "Click the button below to activate your camera. The app will capture a photo automatically.",
            icon="💡",
        )

        if st.button("Activate Camera", use_container_width=True):
            captured_path = take_photo(IMAGES_DIR)
            if captured_path:
                st.session_state.image_path = captured_path
                st.session_state.analysis_complete = False
                st.rerun()

# --- Step 2: Analyze Scene (This section now runs if an image exists) ---
if st.session_state.image_path:
    st.subheader("Step 2: Review and Analyze")
    try:
        image = Image.open(st.session_state.image_path)
        st.image(image, caption="Selected Scene", use_container_width=True)

        if st.button("🧠 Analyze Scene", use_container_width=True):
            with st.spinner(
                "AI Model is analyzing the scene... This may take a moment."
            ):
                try:
                    uploaded_file = genai.upload_file(path=st.session_state.image_path)
                    response = MODEL.generate_content([PROMPT, uploaded_file])
                    st.session_state.description = response.text
                    tts = gTTS(st.session_state.description)
                    tts_file = os.path.join(TTS_OUTPUT_DIR, "env_description.mp3")
                    tts.save(tts_file)
                    st.session_state.tts_path = tts_file
                    st.session_state.analysis_complete = True
                    genai.delete_file(uploaded_file.name)
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

    except FileNotFoundError:
        st.error("Image file not found. Please upload or capture a new one.")
        reset_app_state()

# --- Step 3: Display Results ---
if st.session_state.analysis_complete:
    st.divider()
    st.subheader("Step 3: Results")
    st.markdown("### 📝 Environment's Description")
    st.write(st.session_state.description)
    st.markdown("### 🔊 Audio Playback")
    st.audio(st.session_state.tts_path, format="audio/mp3", loop=True, autoplay=True)
