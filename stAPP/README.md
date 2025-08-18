### 👁️ SightGuide Assistant

SightGuide is an interactive web application built with Streamlit designed to assist visually impaired individuals. The application analyzes an image of the user's surroundings—either uploaded or captured live via a webcam—and uses the Google Gemini AI model to provide a concise, audio-visual description of the scene, with a focus on identifying potential obstacles.

(Note: You should replace the link above with a real screenshot of your running application.)
### ✨ Features
- Dual Image Input: Users can either upload an image file (JPG, PNG) or take a live photo using their webcam.
- Sample Image: Includes a one-click sample image for quick testing and demonstration.
- AI-Powered Scene Analysis: Leverages the powerful Google Gemini model to understand the content of the image from a first-person perspective.
- Obstacle-Focused Descriptions: The AI prompt is specifically engineered to describe the scene and highlight immediate obstacles.
- Text-to-Speech (TTS) Output: Generates and plays an audio version of the description for easy accessibility.
- Clean & Interactive UI: Built with Streamlit for a responsive and user-friendly experience.

### 📋 Prerequisites
Before you begin, ensure you have the following installed and configured:
1. Python 3.11+: This project requires a modern version of Python.
2. uv: A fast Python package installer and resolver. If you don't have it, install it via: `pipx install uv` or if you don't have pipx `pip install uv`

3. Google API Key: You need a Google API key with access to the Gemini models. You can get one for free from Google AI Studio.

#### 🚀 Installation & Setup
Follow these steps to get the application running locally.

##### Clone the Repository
Clone this project to your local machine.

`git clone https://github.com/arslansaeed/MobileApp.git`

#### Create and Activate the Virtual Environment 

Using uv, create a virtual environment. This command will create a .venv folder in your project directory.
`uv venv -p 3.11` or `uv sync`

#### Next, activate the environment. The command differs based on your operating system:

On Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`

On macOS and Linux (Bash/Zsh): `source .venv/bin/activate`

#### Configure Your API Key

    make sure you have your API key in `.env` file

    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"

Replace "YOUR_GOOGLE_API_KEY_HERE" with your actual key.

▶️ Running the Application

With the environment activated and the API key configured, you are ready to run the app.

Execute the following command in your terminal:

`uv run streamlit run stApp/app.py`

Streamlit will start the server and automatically open the application in your default web browser.

📁 Structure

    stAPP/
    │ 
    ├── images/                 # Temporary directory for captured/uploaded images
    ├── tts_output/             # Temporary directory for generated audio files
    ├── app.py                  # The main Streamlit application script
    ├── utils                   # utility folder
    └── README.md               # This file