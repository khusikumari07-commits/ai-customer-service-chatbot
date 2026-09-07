from pathlib import Path
import xml.etree.ElementTree as ET
import re
import math


# Location of the MedQuAD dataset
DATA_PATH = Path("data/MedQuAD-master")


def load_medical_data():
    qa_pairs = []

    for xml_file in DATA_PATH.rglob("*.xml"):
        try:
            root = ET.parse(xml_file).getroot()

            for qa_pair in root.findall(".//QAPair"):
                question = qa_pair.findtext("Question")
                answer = qa_pair.findtext("Answer")

                if question and answer:
                    qa_pairs.append(
                        {
                            "question": question.strip(),
                            "answer": answer.strip()
                        }
                    )

        except (ET.ParseError, OSError):
            continue

    return qa_pairs


def clean_words(text):
    stop_words = {
        "what", "are", "the", "is", "of", "a", "an",
        "do", "does", "how", "can", "to", "for", "in",
        "on", "with", "and", "or", "my", "i", "you",
        "your", "me", "tell", "about", "please",
        "there", "any", "give", "list", "have",
        "available", "could", "would", "should"
    }

    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    return [
        word for word in words
        if word not in stop_words
    ]


def search_medical_data(query, data, top_k=1):
    if not query or not data:
        return None

    query_words = clean_words(query)

    if not query_words:
        return None

    query_set = set(query_words)

    # Words that describe the type of medical question
    question_type_words = {
        "symptoms",
        "symptom",
        "treatment",
        "treat",
        "causes",
        "cause",
        "diagnosis",
        "diagnose",
        "prevention",
        "prevent",
        "risk"
    }

    # Separate the actual medical topic from question-type words
    topic_words = query_set - question_type_words

    best_result = None
    best_score = 0

    for item in data:

        question = item.get("question", "")
        answer = item.get("answer", "")

        if not question or not answer:
            continue

        question_words = clean_words(question)

        if not question_words:
            continue

        question_set = set(question_words)

        # Find words that match between the user query
        # and the medical question
        common_words = query_set.intersection(question_set)

        # Find actual medical-topic matches
        topic_matches = topic_words.intersection(question_set)

        # If the user mentioned a specific topic such as
        # fever, headache, diabetes, asthma, etc.,
        # ignore questions that do not contain that topic.
        if topic_words and not topic_matches:
            continue

        score = 0

        # Score topic matches strongly
        for word in topic_matches:

            if len(word) >= 5:
                score += 10
            else:
                score += 4

        # Score question-type matches
        type_matches = question_type_words.intersection(query_set)

        for word in type_matches:

            if word in question_set:
                score += 6

        # Reward multiple matching words
        if len(common_words) >= 2:
            score += len(common_words) * 5

        # Reward questions that contain most of the topic words
        if topic_words:

            topic_coverage = (
                len(topic_matches) / len(topic_words)
            )

            score += topic_coverage * 15

        # Penalize extremely long questions
        length_penalty = math.sqrt(
            max(len(question_words), 1)
        )

        score = score / length_penalty

        if score > best_score:

            best_score = score
            best_result = item

    if best_result and best_score > 0:
        return best_result["answer"]

    return None


def detect_medical_entities(text):

    text = text.lower()

    symptom_words = [
        "fever",
        "headache",
        "cough",
        "pain",
        "fatigue",
        "tired",
        "nausea",
        "vomiting",
        "dizziness",
        "rash",
        "swelling",
        "weakness",
        "thirst",
        "weight loss",
        "blurred vision"
    ]

    disease_words = [
        "diabetes",
        "asthma",
        "cancer",
        "pneumonia",
        "arthritis",
        "migraine",
        "depression",
        "anemia",
        "influenza",
        "flu",
        "covid",
        "hypertension",
        "heart disease"
    ]

    treatment_words = [
        "medicine",
        "medication",
        "treatment",
        "therapy",
        "surgery",
        "antibiotic",
        "insulin",
        "vaccine"
    ]

    entities = {
        "Symptoms": [],
        "Diseases": [],
        "Treatments": []
    }

    for word in symptom_words:

        if word in text:
            entities["Symptoms"].append(word)

    for word in disease_words:

        if word in text:
            entities["Diseases"].append(word)

    for word in treatment_words:

        if word in text:
            entities["Treatments"].append(word)

    return entities