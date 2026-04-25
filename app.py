import re
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "hi", "hello", "hey", "good morning", "good evening",
                "ok", "okay", "yes", "yes please"
            ],
            "responses": [
                "Hello! How can I help you today?",
                "Hi there, how may I assist you?",
                "Sure, please tell me how I can help you."
            ]
        },
        {
            "tag": "help",
            "patterns": [
                "i need help", "i need information", "help me",
                "can you help", "i want some information", "need support"
            ],
            "responses": [
                "Sure! Please tell me what you need help with.",
                "I’m here to help. Ask me anything about orders, refunds, or timings.",
                "Of course. Please share your question and I will assist you."
            ]
        },
        {
            "tag": "store_hours",
            "patterns": [
                "what are your timings",
                "store open time",
                "when do you close",
                "opening hours",
                "what time are you open",
                "store timings"
            ],
            "responses": [
                "Our support is available 9 AM to 9 PM, Monday to Saturday."
            ]
        },
        {
            "tag": "order_status",
            "patterns": [
                "track my order",
                "order status",
                "where is my order",
                "check my order",
                "can you track my order"
            ],
            "responses": [
                "Please share your order ID and I will help you track it."
            ]
        },
        {
            "tag": "refund_policy",
            "patterns": [
                "refund policy",
                "return product",
                "get my money back",
                "i want refund",
                "how to return product"
            ],
            "responses": [
                "You can return products within 7 days for a full refund, subject to our return conditions."
            ]
        },
        {
            "tag": "goodbye",
            "patterns": [
                "bye", "goodbye", "see you", "talk to you later", "thank you bye"
            ],
            "responses": [
                "Thank you for contacting us. Have a great day!"
            ]
        },
        {
            "tag": "fallback",
            "patterns": [],
            "responses": [
                "I'm not sure I understand. I can connect you to a human agent for better help."
            ]
        }
    ]
}

texts = []
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

clf = MultinomialNB()
clf.fit(X, labels)

def get_intent_rule_based(user_input):
    text = user_input.lower().strip()

    if re.search(r"\b(hi|hello|hey|good morning|good evening|ok|okay|yes)\b", text):
        return "greeting"
    if re.search(r"\b(help|information|support)\b", text):
        return "help"
    if re.search(r"\b(hours|timings|open|close|opening)\b", text):
        return "store_hours"
    if re.search(r"\b(order status|track my order|where is my order|check my order)\b", text):
        return "order_status"
    if re.search(r"\b(refund|return|money back)\b", text):
        return "refund_policy"
    if re.search(r"\b(bye|goodbye|see you)\b", text):
        return "goodbye"

    return "fallback"

def get_intent_ml(user_input):
    X_input = vectorizer.transform([user_input])
    pred = clf.predict(X_input)[0]
    prob = max(clf.predict_proba(X_input)[0])

    if prob < 0.4:
        return get_intent_rule_based(user_input)

    return pred

def get_response(intent):
    for intent_data in intents["intents"]:
        if intent_data["tag"] == intent:
            return random.choice(intent_data["responses"])
    return "Sorry, something went wrong."

def chat_console():
    print("Customer Support Bot (Console): Type 'quit' to exit.")
    print("Using ML intent classifier...")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break

        intent = get_intent_ml(user_input)
        print("Bot:", get_response(intent))

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    user_message = request.json.get("message", "")
    intent = get_intent_ml(user_message)
    bot_reply = get_response(intent)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    print("Customer Chatbot App Started!")
    print("1. Web app: http://127.0.0.1:5000")
    print("2. Console mode: python app.py --console")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--console":
        chat_console()
    else:
        app.run(debug=True, host="127.0.0.1", port=5000)