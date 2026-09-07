import streamlit as st

from sentiment import analyze_sentiment

from medical_qa import (
    load_medical_data,
    search_medical_data,
    detect_medical_entities
)


# Load the medical knowledge base
medical_data = load_medical_data()


# Page settings
st.set_page_config(
    page_title="AI Customer Service Chatbot",
    page_icon="🤖"
)


# Title
st.title("🤖 AI Customer Service Chatbot")

st.write(
    "Chat with the chatbot using sentiment analysis "
    "and medical knowledge."
)


# Show medical dataset information
st.write(
    f"Medical knowledge base loaded: {len(medical_data)} Q&A pairs"
)


# Create chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# User input
user_input = st.chat_input("Type your message...")


if user_input:

    # Save and display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)


    # Detect medical entities
    entities = detect_medical_entities(user_input)


    # Search the medical knowledge base
    medical_result = search_medical_data(
        user_input,
        medical_data
    )


    # If medical information is found
    if medical_result:

        response = (
            "Here is information from the medical knowledge base:\n\n"
            + medical_result
        )


        # Show detected medical entities
        detected_entities = []

        for category, words in entities.items():

            if words:

                detected_entities.append(
                    f"{category}: " + ", ".join(words)
                )


        if detected_entities:

            response += (
                "\n\n**Detected medical information:**\n"
                + "\n".join(detected_entities)
            )


    # If medical information is not found
    else:

        # Analyze sentiment
        sentiment, score = analyze_sentiment(user_input)


        if sentiment == "positive":

            response = (
                "Thank you! 😊 I'm glad to hear that. "
                "How can I assist you further?"
            )


        elif sentiment == "negative":

            response = (
                "I'm sorry you're having a difficult experience. "
                "😔 Please tell me how I can help."
            )


        else:

            response = (
                "Thank you for your message. "
                "How can I assist you?"
            )


        # Display sentiment information
        response += (
            f"\n\nSentiment: {sentiment} "
            f"(confidence: {score:.2f})"
        )


    # Save chatbot response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )


    # Display chatbot response
    with st.chat_message("assistant"):
        st.write(response)