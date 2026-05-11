from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime
from flask import render_template

import os
import re
import nltk
import requests
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# LOAD ENV


load_dotenv()


# DOWNLOAD NLTK


nltk.download("stopwords", quiet=True)


# FLASK


app = Flask(__name__)
CORS(app)


# CONFIG


MODEL_PATH = "YOUR_HF_USERNAME/truthlens-bert"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

current_year = datetime.now().year
print(current_year)
current_date = datetime.now().strftime("%Y-%m-%d")
print(current_date)


# LOAD BERT MODEL


MODEL_LOADED = False

try:

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.eval()

    MODEL_LOADED = True

    print("BERT model loaded successfully")

except Exception as e:

    print(f"Could not load BERT model: {e}")


# SIGNAL DETECTOR


def detect_signals(text):

    t = text.lower()

    signals = []

    fake_patterns = [
        (r"!{2,}", "Multiple exclamation marks"),
        (r"[A-Z]{5,}", "Excessive capitals"),
        (r"\bshocking\b|\bexposed\b|\bbreaking\b", "Sensational keywords"),
        (r"secret|cover.?up|conspiracy", "Conspiracy language"),
        (r"share this|forward this", "Viral call to action"),
        (r"free money|free coin|giveaway", "Unrealistic giveaway"),
    ]

    real_patterns = [
        (r"according to|confirmed|announced", "Proper attribution"),
        (r"study|research|report", "Research references"),
        (r"official|government|committee", "Official source"),
    ]

    for pattern, label in fake_patterns:

        if re.search(pattern, t):

            signals.append({
                "text": label,
                "type": "fake"
            })

    for pattern, label in real_patterns:

        if re.search(pattern, t):

            signals.append({
                "text": label,
                "type": "real"
            })

    if not signals:

        signals.append({
            "text": "No strong signals",
            "type": "neutral"
        })

    return signals


# BERT PREDICTION


def predict_bert(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    pred_class = torch.argmax(probs, dim=1).item()

    confidence = float(probs[0][pred_class]) * 100

    return pred_class, confidence


# OPENROUTER FACT CHECK


def fact_check_with_openrouter(text):

    try:

        if not OPENROUTER_API_KEY:

            return {
                "available": False,
                "response": "Missing OpenRouter API key"
            }

        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        prompt = f"""
Analyze the following news/text and determine if it appears REAL or FAKE.

Today's date is {current_date}.
search for current information up to {current_year}.
Text:
{text}

Provide:
1. Verdict
2. Reason
3. Misinformation signs
4. Short explanation

Keep response concise.

"""

        payload = {
            "model": "google/gemini-3.1-flash-lite",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional fake news detection assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        data = response.json()

        if "choices" not in data:

            return {
                "available": False,
                "response": data
            }

        result = data["choices"][0]["message"]["content"]

        return {
            "available": True,
            "response": result
        }

    except Exception as e:

        return {
            "available": False,
            "response": str(e)
        }


# ROUTES


@app.route("/")
def home():

    return render_template("index.html")



@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "model_type": "BERT + OpenRouter" if MODEL_LOADED else "None"
    })



@app.route("/predict", methods=["POST"])
def predict():

    try:


        data = request.get_json()

        if not data or "text" not in data:


            return jsonify({
                "error": "Missing text field"
            }), 400

        text = data["text"].strip()

        if not text:

            return jsonify({
                "error": "Empty text"
            }), 400

        
        # SIGNAL DETECTION
        

        signals = detect_signals(text)

        
        # BERT PREDICTION
        

        if MODEL_LOADED:


            pred_class, confidence = predict_bert(text)

        else:


            pred_class = 0
            confidence = 50.0

        
        # OPENROUTER FACT CHECK
        

        fact_check = fact_check_with_openrouter(text)

        
        # RESPONSE
        

        
        # FINAL DECISION


        fake_signal_count = len([
        s for s in signals if s["type"] == "fake"
        ])


        ai_response = str(fact_check.get("response", "")).lower()

        ai_says_fake = any(word in ai_response for word in [
        "fake",
        "fabricated",
        "misleading",
        "no credible evidence",
        "false claim",
        "rumor"
        ])

        ai_says_real = ( not ai_says_fake and any(word in ai_response for word in [
        "real",
        "factual",
        "true",
        "accurate",
        "credible",
        "confirmed"
        ]))



        if ai_says_real:

            final_label = "REAL"
            final_prediction = 1
            

        elif ai_says_fake:

            final_label = "FAKE"
            final_prediction = 0
            
            
     


            
        else:

            if pred_class == 0 or fake_signal_count >= 2:
                final_label = "FAKE"
                final_prediction = 0
                
                

            else:
            
                final_label = "REAL"
                final_prediction = 1
                
                



        return jsonify({

            "prediction": final_prediction,

            "label": final_label,

            "bert_prediction": pred_class,

            "bert_label": "FAKE" if pred_class == 0 else "REAL",

            "confidence": round(confidence, 2),

            "signals": signals,

            "fact_check": fact_check

        })

    except Exception as e:
        



        return jsonify({
            "error": str(e)
        }), 500


# MAIN


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("BERT + OpenRouter Fake News Detector")
    print("=" * 60)
    print(f"Model loaded: {MODEL_LOADED}")
    print("URL: http://localhost:7860")
    print("=" * 60 + "\n")

    port = int(os.environ.get("PORT", 7860))

    app.run(debug=False, host="0.0.0.0", port=port)