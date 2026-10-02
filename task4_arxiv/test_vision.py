import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("ERROR: OPENROUTER_API_KEY was not found.")
    raise SystemExit

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "openrouter/free",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe what you can see in this image in a few simple sentences."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://live.staticflickr.com/3851/14825276609_098cac593d_b.jpg"
                    }
                }
            ]
        }
    ]
}

response = requests.post(
    url,
    headers=headers,
    json=data,
    timeout=120
)

print("Status code:", response.status_code)

if response.ok:
    result = response.json()
    print("\nVision model response:")
    print(result["choices"][0]["message"]["content"])
else:
    print("\nOpenRouter error:")
    print(response.text)