
from flask import Flask, request, jsonify, render_template, session
from transformers import pipeline

# from dotenv import load_dotenv
import os
import openai
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_md")  # Use medium-sized model for better word embeddings


# load_dotenv()  # take environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


app = Flask(__name__)
app.secret_key = 'super_secret_key'


sentiment_analysis = pipeline('sentiment-analysis')
ner_pipeline = pipeline('ner', aggregation_strategy="simple")  # Named Entity Recognition


app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")

session_history = []  # Store conversation history here
@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg')
    responses = []
    response = ""

    # # Critical keywords
    # critical_keywords = ["suicide", "harm", "depressed", "anxiety", "panic attack"]
    #
    # # Check for critical mental health issues
    # if any(keyword in user_input.lower() for keyword in critical_keywords):
    #     return jsonify({
    #         'response': 'It seems you are going through something difficult. Please talk to a professional. Here is a helpline: [Helpline URL]'
    #     })

    # Check for sensitive topics using enhanced NLP
    if check_sensitive_topics(user_input):
        response = "<br>" + handle_sensitive()

    # Append the user's message to the session history
    session_history.append({"role": "user", "content": user_input})

    # Analyze mood based on the last user message
    mood_response = analyze_mood()
    if mood_response:
        response = "<br>" + mood_response

    # Check for recurring issues
    recurring_issue_response = check_recurring_issues()
    if recurring_issue_response:
        response = "<br>" + recurring_issue_response

    # Query GPT-4 with conversation history
    bot_res = openai.chat.completions.create(
        model="gpt-4",
        messages=session_history,
        max_tokens=150,
        temperature=0.7,
    )

    # Extract GPT-4's response and add it to the history
    chatbot_reply = bot_res.choices[0].message.content
    session_history.append({"role": "assistant", "content": chatbot_reply})


    responses.append(chatbot_reply)
    if response != "":
        responses.append(response)

    return jsonify({'responses': responses})

    # return chatbot_reply


def analyze_mood():
    # Extract the last 5 user messages from the session history
    user_messages = [message['content'] for message in reversed(session_history) if message['role'] == 'user'][:5]

    if not user_messages or len(user_messages)<6:
        return ""  # No user messages to analyze

    mood_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}  # Initialize mood counters

    # Analyze sentiment of each user message
    for user_message in user_messages:
        sentiment = sentiment_analysis(user_message)[0]
        mood = sentiment['label']  # Returns 'POSITIVE', 'NEGATIVE', or sometimes 'NEUTRAL'

        # Increment the corresponding mood counter
        if mood in mood_counts:
            mood_counts[mood] += 1

    # Determine the most common mood
    common_mood = max(mood_counts, key=mood_counts.get)

    # Respond based on the most frequent mood
    if common_mood == "POSITIVE":
        return "I'm glad to see you're feeling better lately!"
    elif common_mood == "NEGATIVE":
        return "I'm sensing that you might be feeling down lately. I'm here if you want to talk."

    return ""


def check_recurring_issues():
    user_intents = []

    # Define keywords for various mental health-related intents
    intent_keywords = {
        "anxiety": ["anxiety", "anxious", "panic", "nervous"],
        "depression": ["depressed", "depression", "sad", "down"],
        "stress_related_issues": ["stress", "overwhelmed", "pressure", "burden"],
        "schizophrenia": ["schizophrenia", "hallucinations", "delusions"]
    }
    # Check the last three messages in the conversation history for user intents
    recent_messages = [msg for msg in session_history if msg['role'] == 'user'][-3:]  # Get the last 3 user messages

    # Check the conversation history for frequent mental health-related intents

    for message in recent_messages:
        if message['role'] == 'user':
            user_message = message['content'].lower()  # Convert to lowercase for easier matching
            user_doc = nlp(user_message)  # Process user message with spaCy

            for intent, keywords in intent_keywords.items():
                for keyword in keywords:
                    if keyword in user_message:
                        user_intents.append(intent)
                        continue
                    keyword_doc = nlp(keyword)  # Process keyword with spaCy

                    # Calculate similarity
                    if user_doc.similarity(keyword_doc) > 0.7:  # Adjust threshold as needed
                        user_intents.append(intent)
                        continue  # No need to check other keywords if one matches

    # Check for recurring patterns (e.g., if an intent has occurred more than twice)
    for intent in set(user_intents):
        if user_intents.count(intent) == 3:
            return f"It seems you've been discussing {intent} frequently. If you need help, please reach out to our student counseling center: [counselingcenter@utdallas.edu]"

    return ""


def check_sensitive_topics(message):
    # Define critical keywords
    critical_keywords = ["suicide", "harm", "depressed", "anxiety", "panic attack"]

    # Process critical keywords into SpaCy documents for semantic comparison
    critical_keyword_docs = [nlp(keyword) for keyword in critical_keywords]

    # Use SpaCy to analyze the message
    doc = nlp(message)

    # Extract words of interest (nouns, adjectives, and verbs)
    words_of_interest = [token.text.lower() for token in doc if token.pos_ in {"ADJ", "VERB", "NOUN"}]

    # Check for direct critical keywords in the message (exact match)
    if any(keyword in message.lower() for keyword in critical_keywords):
        return True

    # Check if any extracted words of interest are semantically similar to the critical keywords
    for word in words_of_interest:
        word_doc = nlp(word)  # Process each word of interest as a SpaCy document

        # Compare the word with each critical keyword for semantic similarity
        for keyword_doc in critical_keyword_docs:
            if word_doc.similarity(keyword_doc) > 0.7:  # Similarity threshold (can be adjusted)
                return True  # Return True if a critical word or similar word of interest is found

    return False


def handle_sensitive():
    return "It seems like you're going through something difficult. Please reach out to our student counseling center: [counselingcenter@utdallas.edu]"

if __name__ == "__main__":
    app.run()
