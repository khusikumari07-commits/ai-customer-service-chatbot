import base64
import json
import os
import requests

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------
# Detect image type
# --------------------------------------------------

def get_image_mime_type(image_bytes):
    """
    Detect the uploaded image format from its file signature.
    """

    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"

    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
        return "image/webp"

    return "image/jpeg"


# --------------------------------------------------
# Send request to OpenRouter
# --------------------------------------------------

def send_vision_request(
    api_key,
    image_data_url,
    prompt
):
    """
    Send an image + text request to the vision model.
    """

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

                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url": image_data_url
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

    if not response.ok:
        return None, f"Vision API error: {response.text}"

    try:

        result = response.json()

        answer = (
            result["choices"][0]["message"]["content"]
        )

        if not answer or not answer.strip():
            return None, (
                "The vision model returned an empty response."
            )

        return answer.strip(), None

    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError
    ):

        return None, (
            "The vision model returned an "
            "unexpected response format."
        )


# --------------------------------------------------
# Main multimodal analysis
# --------------------------------------------------

def analyze_image(
    image_bytes,
    user_question,
    conversation_context=None
):
    """
    Analyze an image together with the user's question.

    The process contains:
    1. Visual analysis
    2. Context-aware reasoning
    3. Response validation
    4. Safer final response generation
    """

    # --------------------------------------------------
    # Check API key
    # --------------------------------------------------

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:

        return (
            "Error: OPENROUTER_API_KEY was not found."
        )


    # --------------------------------------------------
    # Check image
    # --------------------------------------------------

    if not image_bytes:

        return (
            "Please upload an image so I can analyze it."
        )


    # --------------------------------------------------
    # Default question
    # --------------------------------------------------

    if (
        not user_question
        or not user_question.strip()
    ):

        user_question = (
            "Describe the image, identify the important "
            "information visible in it, and mention anything "
            "that may need clarification."
        )


    # --------------------------------------------------
    # Convert image to Base64
    # --------------------------------------------------

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")


    # --------------------------------------------------
    # Detect image format
    # --------------------------------------------------

    mime_type = get_image_mime_type(
        image_bytes
    )


    image_data_url = (
        f"data:{mime_type};base64,"
        f"{image_base64}"
    )


    # --------------------------------------------------
    # Conversation context
    # --------------------------------------------------

    context = (
        conversation_context
        if conversation_context
        else "No previous conversation context."
    )


    # ==================================================
    # STEP 1 - Generate visual answer
    # ==================================================

    analysis_prompt = f"""
You are a multimodal AI assistant.

Analyze the uploaded image and answer the user's
question using information that can reasonably be
supported by the image.

USER QUESTION:
{user_question}

PREVIOUS CONVERSATION:
{context}

Instructions:

1. Identify the important visual information first.
2. Answer the user's question clearly.
3. Use previous conversation context when relevant.
4. If the question is ambiguous, explain what is unclear.
5. If the question refers to something that is not
   present in the image, clearly say that.
6. Do not invent objects, people, text, numbers,
   identities, or events that cannot be supported
   by the image.
7. If visible text is relevant, use it carefully.
8. If the image does not contain enough evidence,
   say so.
9. Give a concise explanation when reasoning is useful.
10. Do not claim certainty when the image is unclear.

Generate a draft answer based only on the image,
question, and available conversation context.
"""


    draft_answer, error = send_vision_request(
        api_key,
        image_data_url,
        analysis_prompt
    )


    if error:

        return error


    if not draft_answer:

        return (
            "I could not generate a reliable answer "
            "from the image."
        )


    # ==================================================
    # STEP 2 - Validate the generated answer
    # ==================================================

    validation_prompt = f"""
You are the response-validation component of a
multimodal AI assistant.

Your job is to check whether the DRAFT ANSWER is
supported by the uploaded image and the user's question.

USER QUESTION:
{user_question}

DRAFT ANSWER:
{draft_answer}

Check the draft against the actual image.

Validation rules:

1. Check whether the objects and visual facts in the
   draft are actually supported by the image.
2. Check whether the draft answers the user's question.
3. Check for invented details or unsupported claims.
4. Check whether the draft incorrectly identifies
   something that is not visible.
5. Check whether the response should express uncertainty.
6. If the draft is supported, return it unchanged.
7. If the draft contains unsupported information,
   rewrite it using only information supported by
   the image.
8. Do not add new unsupported information.

Return ONLY valid JSON in this exact format:

{{
    "valid": true,
    "answer": "final answer"
}}

OR:

{{
    "valid": false,
    "answer": "corrected final answer"
}}
"""


    validation_result, error = send_vision_request(
        api_key,
        image_data_url,
        validation_prompt
    )


    # --------------------------------------------------
    # If validation request fails
    # --------------------------------------------------

    if error:

        # The original answer was already generated
        # using evidence-based instructions.
        return draft_answer


    # ==================================================
    # STEP 3 - Read validation result
    # ==================================================

    try:

        # Remove possible markdown code fences
        cleaned_result = (
            validation_result
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        validation_data = json.loads(
            cleaned_result
        )

        final_answer = validation_data.get(
            "answer",
            ""
        )

        is_valid = validation_data.get(
            "valid",
            True
        )


        # --------------------------------------------------
        # Make sure final answer is not empty
        # --------------------------------------------------

        if (
            not final_answer
            or not str(final_answer).strip()
        ):

            return draft_answer


        # --------------------------------------------------
        # Return validated/corrected response
        # --------------------------------------------------

        return str(final_answer).strip()


    except (
        json.JSONDecodeError,
        TypeError,
        AttributeError
    ):

        # If validation output cannot be parsed,
        # safely use the original evidence-based answer.
        return draft_answer