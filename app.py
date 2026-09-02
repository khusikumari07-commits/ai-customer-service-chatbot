import streamlit as st
from sentiment import analyze_sentiment

st.set_page_config(
    page_title="AI Customer Service Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI Customer Service Chatbot")
st.write("Chat with the assistant and get sentiment-aware responses.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    sentiment, score = analyze_sentiment(user_input)

    if sentiment == "positive":
        response = (
            f"Thank you! 😊 I'm glad to hear that. "
            f"(Sentiment: {sentiment}, confidence: {score:.2f})"
        )

    elif sentiment == "negative":
        response = (
            f"I'm sorry you're having a bad experience. "
            f"Let me help you resolve this. "
            f"(Sentiment: {sentiment}, confidence: {score:.2f})"
        )

    else:
        response = (
            f"Thank you for your message. How can I assist you? "
            f"(Sentiment: {sentiment}, confidence: {score:.2f})"
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.write(response)