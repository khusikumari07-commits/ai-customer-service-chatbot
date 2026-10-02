import os
import requests

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------
# Multilingual AI Assistant
# --------------------------------------------------

def multilingual_response(
    user_message,
    conversation_context=None
):
    """
    Detect the language of the CURRENT user message
    and respond in that language while preserving
    the meaning and context of the conversation.
    """

    # --------------------------------------------------
    # Get API key
    # --------------------------------------------------

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return "Error: OPENROUTER_API_KEY was not found."


    # --------------------------------------------------
    # Validate message
    # --------------------------------------------------

    if not user_message or not user_message.strip():
        return "Please enter a message."


    # --------------------------------------------------
    # Previous conversation
    # --------------------------------------------------

    context = (
        conversation_context
        if conversation_context
        else "No previous conversation."
    )


    # --------------------------------------------------
    # Multilingual prompt
    # --------------------------------------------------

    prompt = f"""
You are a multilingual AI customer service assistant.

CURRENT USER MESSAGE:
{user_message}

PREVIOUS CONVERSATION:
{context}

Your most important rule:

THE LANGUAGE OF THE CURRENT USER MESSAGE HAS PRIORITY
OVER THE LANGUAGE USED IN THE PREVIOUS CONVERSATION.

Follow these rules carefully:

1. First identify the language of the CURRENT user message.

2. Respond primarily in the language of the CURRENT user message.

3. Previous conversation is provided ONLY to understand
   context, topic, references, and user intent.

4. NEVER change the response language merely because
   an earlier message was written in another language.

5. If the conversation changes from English to Hindi,
   answer the new Hindi message in Hindi.

6. If the conversation changes from Hindi back to English,
   answer the new English message in English.

7. Support at least:
   - English
   - Hindi
   - Spanish
   - French

8. Support mixed-language messages such as Hinglish.
   If the message is mixed Hindi-English, understand the
   complete meaning and respond naturally in a similar
   mixed style or predominantly Hindi unless the user
   clearly requests another language.

9. Preserve the original topic and intent when the user
   changes language.

10. If the user asks a follow-up such as:
    "Give me an example"
    use the previous conversation to understand what
    "it" refers to.

11. If the current message is ambiguous, ask for
    clarification in the language of the current message.

12. If the user explicitly requests a particular output
    language, follow that request.

13. Do not translate the previous conversation unless
    the user asks you to.

14. Do not invent information.

15. Keep the answer natural, clear, and useful.

IMPORTANT:
Determine the response language from the CURRENT USER
MESSAGE first. Context must not override the current
message language.

Now answer the CURRENT USER MESSAGE.
"""


    # --------------------------------------------------
    # OpenRouter API
    # --------------------------------------------------

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "qwen/qwen3-vl-8b-instruct",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }


    # --------------------------------------------------
    # API request
    # --------------------------------------------------

    try:

        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=120
        )


        # --------------------------------------------------
        # API error
        # --------------------------------------------------

        if not response.ok:

            return (
                f"Multilingual API error: "
                f"{response.text}"
            )


        # --------------------------------------------------
        # Read response
        # --------------------------------------------------

        result = response.json()

        answer = (
            result["choices"][0]["message"]["content"]
        )


        if not answer or not answer.strip():

            return (
                "The multilingual model returned "
                "an empty response."
            )


        return answer.strip()


    except requests.RequestException as error:

        return (
            "Connection error while using the "
            f"multilingual assistant: {error}"
        )


    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError
    ):

        return (
            "The multilingual model returned "
            "an unexpected response format."
        )