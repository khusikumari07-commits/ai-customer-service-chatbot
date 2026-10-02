import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

URL = "https://openrouter.ai/api/v1/chat/completions"


def explain_paper(title, abstract):
    """Use an open-source LLM through OpenRouter to explain a paper."""

    if not API_KEY:
        return "OpenRouter API key was not found."

    prompt = f"""
You are an expert Computer Science research assistant.

Research paper:
Title: {title}

Abstract:
{abstract}

Explain this research paper in simple language.

Give the answer in this format:

1. Main Idea
2. Problem Addressed
3. Method Used
4. Important Concepts
5. Simple Explanation
6. Possible Applications

Do not invent information that is not supported by the abstract.
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openrouter/free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(
            URL,
            headers=headers,
            json=data,
            timeout=60
        )

        if response.ok:
            result = response.json()
            return result["choices"][0]["message"]["content"]

        return f"OpenRouter error: {response.text}"

    except Exception as error:
        return f"Connection error: {error}"


if __name__ == "__main__":

    title = input("\nEnter paper title: ").strip()

    abstract = input(
        "\nEnter paper abstract: "
    ).strip()

    print("\nGenerating explanation...\n")

    answer = explain_paper(title, abstract)

    print(answer)