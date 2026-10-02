import sys
from pathlib import Path

# Allow this test file to import modules from the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import requests
from multimodal_ai import analyze_image


# Test image
image_url = "https://live.staticflickr.com/3851/14825276609_098cac593d_b.jpg"


# Download the test image
image_response = requests.get(image_url, timeout=30)
image_response.raise_for_status()


# User's question about the image
question = "What can you see in this image?"


# Send image and question to our Task 5 module
answer = analyze_image(
    image_response.content,
    question
)


# Display the result
print("\nMultimodal AI response:")
print(answer)