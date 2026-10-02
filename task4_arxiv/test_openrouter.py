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
            "content": "Explain artificial intelligence in one simple sentence."
        }
    ]
}

response = requests.post(url, headers=headers, json=data, timeout=60)

print("Status code:", response.status_code)

if response.ok:
    result = response.json()
    print("\nOpenRouter response:")
    print(result["choices"][0]["message"]["content"])
else:
    print("\nOpenRouter error:")
    print(response.text)