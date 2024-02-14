# VisionAI

## Introduction
VisionAI is a Streamlit application that describes user-provided images using Azure's GPT-4 Vision API. It offers an interactive UI where users can upload an image and receive an image description.

## Prerequisites
- Python 3.8 or higher
- An Azure account with access to the DALL-E 3 API
- API key and endpoint from Azure for the DALL-E 3 service

## Setup
1. **Clone the repository**:
```
git clone https://github.com/EsteMac/VisionAI.git
cd VisionAI
```

2. **Create a virtual environment** (optional but recommended):
```
python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate
```

3. **Install required packages**:
pip install -r requirements.txt

4. **Set up environment variables**:
- Create a `.env` file in the project directory.
- Add your Azure GPT-4 Vision API key and endpoint to the `.env` file:
  ```
  AZURE_OPENAI_API_GPT4V_KEY=your_api_key_here
  AZURE_OPENAI_API_ENDPOINT_GPT4V=your_api_endpoint_here
  ```

## Usage
Run the Streamlit application with:
```
streamlit run main.py
```

Navigate to the provided URL in your web browser to interact with the application. You can upload an image, and generate image descriptions based on your image. 

## Features
- **Interactive UI**: Streamlit-based UI for an intuitive user experience.

## Disclaimer
This project is for educational and demonstration purposes only. Please adhere to Azure's API usage policies and guidelines.


