import os
from openai import AzureOpenAI  # Import client for Azure OpenAI
import streamlit as st
from dotenv import load_dotenv
import base64
from mimetypes import guess_type
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()  # Take environment variables from .env.

# Get API key and endpoint from environment variable
AZURE_OPENAI_API_GPT4V_KEY = os.getenv("AZURE_OPENAI_API_GPT4V_KEY")
AZURE_OPENAI_API_ENDPOINT_GPT4V = os.getenv("AZURE_OPENAI_API_ENDPOINT_GPT4V")

# Function to convert image bytes into a base64 data URL
def image_bytes_to_data_url(image_bytes, mime_type='image/jpeg'):
    base64_encoded_data = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

# Function to query Azure GPT-4-V with an image prompt
def query_azure_gpt4v_with_image(image_bytes, api_key, api_endpoint):
    client = AzureOpenAI(
        api_key=api_key,  
        api_version="2023-07-01-preview",
        azure_endpoint=api_endpoint
    )

    # Convert image bytes to a base64 data URL
    data_url = image_bytes_to_data_url(image_bytes)
    
    # Make the API call
    try:
        response = client.chat.completions.create(
            model="gpt-4.0-vision-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at identifying objects, people, and places in images."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this picture:"},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]}
            ],
            max_tokens=2000
        )

        if response.choices:
            # Directly access the content attribute of the message object
            assistant_message = response.choices[0].message.content  # Updated to use dot notation
            return assistant_message
        else:
            return "No description provided."
    except Exception as e:
        logging.error(f"Error calling Azure OpenAI: {str(e)}")
        return "An error occurred while processing your request."

# Main Streamlit app
def main():
    st.title("Image Description with Azure GPT-4 Vision")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        
        # Query Azure GPT-4-V and display the result
        description = query_azure_gpt4v_with_image(image_bytes, AZURE_OPENAI_API_GPT4V_KEY, AZURE_OPENAI_API_ENDPOINT_GPT4V)
        st.write(description)

if __name__ == "__main__":
    main()