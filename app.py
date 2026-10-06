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

from arxiv_expert import (
    load_papers,
    search_papers,
    extract_concepts,
    create_concept_graph,
    ask_openrouter
)


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
# Load arXiv knowledge base
# --------------------------------------------------

@st.cache_data
def get_arxiv_papers():
    return load_papers()


arxiv_papers = get_arxiv_papers()


# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title("🤖 AI Customer Service Chatbot")

st.write(
    "Chat with the assistant using sentiment analysis, "
    "medical knowledge, dynamically updated knowledge, "
    "ArXiv research assistance, multimodal image analysis, "
    "and multilingual support."
)

st.write(
    f"Medical knowledge base loaded: "
    f"{len(medical_data)} Q&A pairs"
)

st.write(
    f"ArXiv Computer Science papers loaded: "
    f"{len(arxiv_papers)}"
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

        image_bytes = uploaded_image.getvalue()

        with st.spinner(
            "Analyzing the image..."
        ):

            image_response = analyze_image(
                image_bytes,
                image_question,
                conversation_context
            )

        st.markdown("### 🤖 Image Analysis")

        st.write(image_response)

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
# TASK 4 - ARXIV COMPUTER SCIENCE EXPERT
# ==================================================

st.divider()

st.subheader(
    "📚 Task 4 — ArXiv Computer Science Expert"
)

st.write(
    "Search Computer Science research papers, "
    "explore extracted concepts, generate AI explanations, "
    "and ask follow-up questions about selected papers."
)


# --------------------------------------------------
# Paper search
# --------------------------------------------------

arxiv_query = st.text_input(
    "🔎 Search Computer Science research papers",
    placeholder=(
        "Example: transformer models for natural "
        "language processing"
    ),
    key="arxiv_search_query"
)


if arxiv_query:

    arxiv_results = search_papers(
        arxiv_query,
        top_k=5
    )

    if not arxiv_results:

        st.warning(
            "No matching Computer Science papers found."
        )

    else:

        st.markdown("### 📄 Research Paper Results")

        for number, result in enumerate(
            arxiv_results,
            start=1
        ):

            paper = result["paper"]

            title = paper.get(
                "title",
                "Untitled"
            ).strip()

            abstract = paper.get(
                "abstract",
                "No abstract available."
            ).strip()

            paper_id = paper.get(
                "id",
                ""
            )

            categories = paper.get(
                "categories",
                ""
            )

            concepts = extract_concepts(
                paper
            )

            with st.expander(
                f"{number}. {title}"
            ):

                st.write(
                    f"**arXiv ID:** {paper_id}"
                )

                st.write(
                    f"**Categories:** {categories}"
                )

                st.write(
                    f"**Relevance Score:** "
                    f"{result['score']:.3f}"
                )

                st.write("### Abstract")

                st.write(abstract)

                st.write(
                    "### 🧠 Extracted Concepts"
                )

                if concepts:

                    st.write(
                        ", ".join(concepts)
                    )

                else:

                    st.write(
                        "No major concepts detected."
                    )


                # ------------------------------------------
                # AI explanation
                # ------------------------------------------

                if st.button(
                    "🤖 Explain this paper",
                    key=f"arxiv_explain_{number}"
                ):

                    prompt = f"""
Explain this Computer Science research paper
in simple but technically accurate language.

Title:
{title}

Abstract:
{abstract}

Extracted concepts:
{", ".join(concepts)}

Please provide:

1. Research problem
2. Main idea
3. Method or approach
4. Important concepts
5. Possible applications
6. Simple explanation for a student

Do not invent details that are not supported
by the supplied paper information.
"""

                    with st.spinner(
                        "Generating explanation..."
                    ):

                        explanation = ask_openrouter(
                            prompt
                        )

                    st.markdown(
                        "### 🤖 AI Explanation"
                    )

                    st.write(explanation)

                    st.session_state[
                        "selected_arxiv_paper"
                    ] = paper


# --------------------------------------------------
# Follow-up questions
# --------------------------------------------------

if "selected_arxiv_paper" in st.session_state:

    selected_paper = st.session_state[
        "selected_arxiv_paper"
    ]

    st.markdown(
        "### 💬 Ask Follow-up Questions"
    )

    followup = st.text_input(
        "Ask something about the selected paper",
        placeholder=(
            "Example: What problem does this research solve?"
        ),
        key="arxiv_followup"
    )

    if followup:

        title = selected_paper.get(
            "title",
            ""
        ).strip()

        abstract = selected_paper.get(
            "abstract",
            ""
        ).strip()

        concepts = extract_concepts(
            selected_paper
        )

        prompt = f"""
Answer the user's follow-up question
using the selected research paper as context.

Paper title:
{title}

Paper abstract:
{abstract}

Research concepts:
{", ".join(concepts)}

User question:
{followup}

Answer clearly and technically accurately.

Do not invent information that is not
supported by the paper.
"""

        with st.spinner(
            "Thinking..."
        ):

            followup_answer = ask_openrouter(
                prompt
            )

        st.write(
            followup_answer
        )


# --------------------------------------------------
# Concept visualization
# --------------------------------------------------

st.markdown(
    "### 🧠 Research Concept Visualization"
)

st.write(
    "Select a Computer Science paper to view "
    "its extracted research concepts."
)


paper_titles = [

    paper.get(
        "title",
        ""
    ).strip()

    for paper in arxiv_papers[:200]

]


if paper_titles:

    selected_title = st.selectbox(
        "Choose a paper",
        paper_titles,
        key="arxiv_visualization_paper"
    )

    selected_paper_for_graph = next(

        (
            paper

            for paper in arxiv_papers

            if paper.get(
                "title",
                ""
            ).strip()
            == selected_title
        ),

        None
    )

    if selected_paper_for_graph:

        concepts = extract_concepts(
            selected_paper_for_graph
        )

        if concepts:

            st.write(
                "**Detected concepts:**"
            )

            st.write(
                " • ".join(concepts)
            )

            if st.button(
                "📊 Show Concept Graph",
                key="show_arxiv_graph"
            ):

                fig = create_concept_graph(
                    selected_paper_for_graph,
                    concepts
                )

                st.pyplot(fig)

                import matplotlib.pyplot as plt

                plt.close(fig)

        else:

            st.write(
                "No major concepts detected."
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


        if detected_parts:

            st.caption(
                "Detected medical information: "
                + " | ".join(
                    detected_parts
                )
            )


        if knowledge_source == "dynamic":

            if dynamic_metadata:

                source = dynamic_metadata[0].get(
                    "source",
                    "Unknown"
                )

                st.caption(
                    f"Knowledge source: {source}"
                )


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