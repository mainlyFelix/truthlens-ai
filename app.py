import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

print("Initializing AI... Downloading robust RoBERTa model. Patience.")
try:
    # Upgrading to a highly generalized, multi-source fine-tuned model
    # to severely reduce formatting and keyword bias.
    classifier = pipeline("text-classification", model="hamzab/roberta-fake-news-classification")
    print("Model loaded successfully. Server is ready.")
except Exception as e:
    print(f"Error loading model: {e}")


def extract_text_from_url(url):
    """Scrapes the main paragraph text from a given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text.strip()
    except Exception:
        return None


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online"}), 200


@app.route('/analyze', methods=['POST'])
def analyze_content():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No payload provided."}), 400

    content = ""
    if 'url' in data:
        content = extract_text_from_url(data['url'])
        if not content:
            return jsonify({"error": "Could not extract text from the provided URL."}), 400
    elif 'text' in data:
        content = data['text']
    else:
        return jsonify({"error": "Invalid payload format."}), 400

    if len(content) < 15:
        return jsonify({"error": "Content too short for accurate analysis."}), 400

    truncated_content = content[:2500]

    try:
        # Run inference
        prediction = classifier(truncated_content)[0]
        label = prediction['label'].lower()
        confidence = round(prediction['score'] * 100)

        # For this model: LABEL_0 is Fake, LABEL_1 is Real
        if "0" in label or "fake" in label:
            verdict = "fake"
        elif "1" in label or "real" in label or "true" in label:
            verdict = "real"
        else:
            verdict = "inconclusive"

        return jsonify({
            "verdict": verdict,
            "confidence": confidence,
            "text_preview": content[:150] + "..." if len(content) > 150 else content
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)