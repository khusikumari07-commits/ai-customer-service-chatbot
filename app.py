import streamlit as st

from sentiment import analyze_sentiment

from medical_qa import (
    load_medical_data,
    search_medical_data,
    detect_medical_entities
)

from knowledge_search import search_knowledge


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
    "Chat with the chatbot using sentiment analysis, "
    "medical knowledge, and dynamically updated knowledge."
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


# --------------------------------------------------
# User input
# --------------------------------------------------

user_input = st.chat_input("Type your message...")


if user_input:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)


    # --------------------------------------------------
    # Detect medical information
    # --------------------------------------------------

    medical_entities = detect_medical_entities(user_input)

    detected_parts = []

    if medical_entities.get("Symptoms"):
        detected_parts.append(
            "Symptoms: "
            + ", ".join(medical_entities["Symptoms"])
        )

    if medical_entities.get("Diseases"):
        detected_parts.append(
            "Diseases: "
            + ", ".join(medical_entities["Diseases"])
        )

    if medical_entities.get("Treatments"):
        detected_parts.append(
            "Treatments: "
            + ", ".join(medical_entities["Treatments"])
        )


    # --------------------------------------------------
    # Analyze sentiment
    # --------------------------------------------------

    sentiment_result = analyze_sentiment(user_input)

    sentiment = "neutral"

    if isinstance(sentiment_result, tuple):
        if len(sentiment_result) > 0:
            sentiment = str(
                sentiment_result[0]
            ).lower()

    elif isinstance(sentiment_result, dict):
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
    # Response selection
    # --------------------------------------------------

    response = ""
    knowledge_source = ""


    # Medical questions get medical knowledge
    if detected_parts:

        medical_results = search_medical_data(
            user_input,
            medical_data
        )

        if medical_results:

            if isinstance(medical_results, list):
                response = medical_results[0]
            else:
                response = str(medical_results)

            knowledge_source = "medical"


    # Positive customer messages get sentiment response
    if not response and "positive" in sentiment:

        response = (
            "I'm glad to hear that! 😊 "
            "How can I help you further?"
        )

        knowledge_source = "sentiment"


    # Negative customer messages get sentiment response
    elif not response and "negative" in sentiment:

        response = (
            "I'm sorry you're having a difficult "
            "experience. I'll do my best to help."
        )

        knowledge_source = "sentiment"


    # Neutral questions use the dynamic knowledge base
    if not response:

        try:

            dynamic_documents, dynamic_metadata = search_knowledge(
                user_input,
                results=3
            )

            if dynamic_documents:

                response = (
                    "Here is information from the "
                    "dynamically updated knowledge base:\n\n"
                    + dynamic_documents[0]
                )

                knowledge_source = "dynamic"

                if dynamic_metadata:
                    source = dynamic_metadata[0].get(
                        "source",
                        "Unknown"
                    )

            else:

                response = (
                    "Thank you for your message. "
                    "How can I help you?"
                )

        except Exception:

            response = (
                "Thank you for your message. "
                "How can I help you?"
            )


    # --------------------------------------------------
    # Display assistant response
    # --------------------------------------------------

    with st.chat_message("assistant"):

        st.write(response)

        # Show detected medical information
        if detected_parts:

            st.caption(
                "Detected medical information: "
                + " | ".join(detected_parts)
            )

        # Show dynamic knowledge source
        if knowledge_source == "dynamic":

            if dynamic_metadata:

                source = dynamic_metadata[0].get(
                    "source",
                    "Unknown"
                )

                st.caption(
                    f"Knowledge source: {source}"
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
    