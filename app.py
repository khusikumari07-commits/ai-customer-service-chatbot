import streamlit as st

from sentiment import analyze_sentiment

from medical_qa import (
    load_medical_data,
    search_medical_data,
    detect_medical_entities
)

from knowledge_search import search_knowledge

from multimodal_ai import analyze_image

from multilingual_ai import multilingual_response


# --------------------------------------------------
# Page settings
# --------------------------------------------------

st.set_page_config(
    page_title="AI Customer Service Chatbot",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# Load medical knowledge base
# --------------------------------------------------

@st.cache_data
def get_medical_data():
    return load_medical_data()


medical_data = get_medical_data()


# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title("🤖 AI Customer Service Chatbot")

st.write(
    "Chat with the assistant using sentiment analysis, "
    "medical knowledge, dynamically updated knowledge, "
    "multimodal image analysis, and multilingual support."
)

st.write(
    f"Medical knowledge base loaded: "
    f"{len(medical_data)} Q&A pairs"
)


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ==================================================
# TASK 5 - MULTIMODAL AI ASSISTANT
# ==================================================

st.divider()

st.subheader("🖼️ Multimodal AI Assistant")

st.write(
    "Upload an image and ask a question about it."
)


uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    key="multimodal_image"
)


image_question = st.text_input(
    "Ask a question about the image",
    placeholder="Example: What can you see in this image?",
    key="image_question"
)


if uploaded_image:

    st.image(
        uploaded_image,
        caption="Uploaded image",
        use_container_width=True
    )


    if st.button(
        "🔍 Analyze Image",
        key="analyze_image_button"
    ):

        # ----------------------------------------------
        # Build previous conversation context
        # ----------------------------------------------

        previous_messages = []

        for message in st.session_state.messages:

            role = message.get("role", "")
            content = message.get("content", "")

            previous_messages.append(
                f"{role}: {content}"
            )


        conversation_context = "\n".join(
            previous_messages[-6:]
        )


        # ----------------------------------------------
        # Get image
        # ----------------------------------------------

        image_bytes = uploaded_image.getvalue()


        # ----------------------------------------------
        # Analyze image
        # ----------------------------------------------

        with st.spinner(
            "Analyzing the image..."
        ):

            image_response = analyze_image(
                image_bytes,
                image_question,
                conversation_context
            )


        # ----------------------------------------------
        # Display result
        # ----------------------------------------------

        st.markdown("### 🤖 Image Analysis")

        st.write(image_response)


        # ----------------------------------------------
        # Save conversation
        # ----------------------------------------------

        image_user_message = image_question

        if not image_user_message:

            image_user_message = (
                "Please analyze the uploaded image."
            )


        st.session_state.messages.append(
            {
                "role": "user",
                "content": (
                    "🖼️ Image: "
                    + image_user_message
                )
            }
        )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": image_response
            }
        )


# ==================================================
# NORMAL CHAT
# ==================================================

st.divider()

st.subheader("💬 Chat")


user_input = st.chat_input(
    "Type your message..."
)


if user_input:

    # --------------------------------------------------
    # Show user message
    # --------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    with st.chat_message("user"):

        st.write(user_input)


    # --------------------------------------------------
    # Build conversation context
    # --------------------------------------------------

    previous_messages = []

    for message in st.session_state.messages[:-1]:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )

        previous_messages.append(
            f"{role}: {content}"
        )


    conversation_context = "\n".join(
        previous_messages[-8:]
    )


    # --------------------------------------------------
    # Detect medical information
    # --------------------------------------------------

    medical_entities = detect_medical_entities(
        user_input
    )

    detected_parts = []


    if medical_entities.get("Symptoms"):

        detected_parts.append(
            "Symptoms: "
            + ", ".join(
                medical_entities["Symptoms"]
            )
        )


    if medical_entities.get("Diseases"):

        detected_parts.append(
            "Diseases: "
            + ", ".join(
                medical_entities["Diseases"]
            )
        )


    if medical_entities.get("Treatments"):

        detected_parts.append(
            "Treatments: "
            + ", ".join(
                medical_entities["Treatments"]
            )
        )


    # --------------------------------------------------
    # Sentiment analysis
    # --------------------------------------------------

    sentiment_result = analyze_sentiment(
        user_input
    )

    sentiment = "neutral"


    if isinstance(
        sentiment_result,
        tuple
    ):

        if len(sentiment_result) > 0:

            sentiment = str(
                sentiment_result[0]
            ).lower()


    elif isinstance(
        sentiment_result,
        dict
    ):

        sentiment = str(
            sentiment_result.get(
                "label",
                sentiment_result.get(
                    "sentiment",
                    "neutral"
                )
            )
        ).lower()


    else:

        sentiment = str(
            sentiment_result
        ).lower()


    # --------------------------------------------------
    # Find relevant information
    # --------------------------------------------------

    source_information = ""
    knowledge_source = ""
    dynamic_metadata = []


    # --------------------------------------------------
    # Medical knowledge
    # --------------------------------------------------

    if detected_parts:

        medical_results = search_medical_data(
            user_input,
            medical_data
        )


        if medical_results:

            if isinstance(
                medical_results,
                list
            ):

                source_information = str(
                    medical_results[0]
                )

            else:

                source_information = str(
                    medical_results
                )

            knowledge_source = "medical"


    # --------------------------------------------------
    # Dynamic knowledge
    # --------------------------------------------------

    if not source_information:

        try:

            (
                dynamic_documents,
                dynamic_metadata
            ) = search_knowledge(
                user_input,
                results=3
            )


            if dynamic_documents:

                source_information = (
                    dynamic_documents[0]
                )

                knowledge_source = "dynamic"


        except Exception:

            source_information = ""


    # ==================================================
    # TASK 6 - MULTILINGUAL RESPONSE
    # ==================================================

    # We use the multilingual assistant for the final
    # response so the current user's language is preserved.

    if source_information:

        multilingual_context = f"""
Previous conversation:
{conversation_context}

Relevant information retrieved from the chatbot's
knowledge sources:

{source_information}

Detected medical information:
{", ".join(detected_parts) if detected_parts else "None"}

Sentiment:
{sentiment}

Use the retrieved information when answering.
Do not invent facts that are not supported by it.
"""

    else:

        multilingual_context = f"""
Previous conversation:
{conversation_context}

There is no specific retrieved knowledge for this
message.

Detected medical information:
{", ".join(detected_parts) if detected_parts else "None"}

Sentiment:
{sentiment}
"""


    # --------------------------------------------------
    # Generate multilingual response
    # --------------------------------------------------

    response = multilingual_response(
        user_input,
        conversation_context=multilingual_context
    )


    # --------------------------------------------------
    # Safety fallback
    # --------------------------------------------------

    if not response or not response.strip():

        if source_information:

            response = source_information

        elif "positive" in sentiment:

            response = (
                "I'm glad to hear that! 😊 "
                "How can I help you further?"
            )

        elif "negative" in sentiment:

            response = (
                "I'm sorry you're having a difficult "
                "experience. I'll do my best to help."
            )

        else:

            response = (
                "Thank you for your message. "
                "How can I help you?"
            )


    # --------------------------------------------------
    # Display assistant response
    # --------------------------------------------------

    with st.chat_message("assistant"):

        st.write(response)


        # ----------------------------------------------
        # Medical information detected
        # ----------------------------------------------

        if detected_parts:

            st.caption(
                "Detected medical information: "
                + " | ".join(
                    detected_parts
                )
            )


        # ----------------------------------------------
        # Dynamic knowledge source
        # ----------------------------------------------

        if knowledge_source == "dynamic":

            if dynamic_metadata:

                source = dynamic_metadata[0].get(
                    "source",
                    "Unknown"
                )

                st.caption(
                    f"Knowledge source: {source}"
                )


        # ----------------------------------------------
        # Multilingual feature
        # ----------------------------------------------

        st.caption(
            "🌐 Multilingual response enabled"
        )


    # --------------------------------------------------
    # Save assistant response
    # --------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )