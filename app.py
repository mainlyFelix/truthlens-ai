import os
import json
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an expert fact-checker and media literacy analyst. Your job is to assess whether a piece of text is likely real/credible news or fake/misleading content.

Analyze the provided text and respond with ONLY a valid JSON object in this exact format:
{
  "verdict": "real" | "fake" | "misleading" | "satire" | "inconclusive",
  "confidence": <integer 0-100>,
  "reasoning": "<2-3 sentence explanation of your verdict>",
  "red_flags": ["<flag1>", "<flag2>"] 
}

Verdict definitions:
- "real": Credible, factual reporting with verifiable claims and neutral tone
- "fake": Demonstrably false claims, fabricated events, or deliberate misinformation
- "misleading": Contains true elements but framed deceptively, uses selective facts, or has clickbait framing
- "satire": Clearly satirical or parody content not intended to be taken literally
- "inconclusive": Not enough information to make a determination

Red flags to look for (include only those that apply, can be empty list):
- Sensationalist/emotional language
- Vague or missing sources
- Implausible claims
- Conspiracy framing
- Clickbait headline patterns
- Logical fallacies
- Known misinformation narratives

Be fair and balanced. Mainstream news, even if you disagree with its framing, should not be marked fake without strong reason. 
Only respond with the JSON object, no other text."""


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
    return jsonify({"status": "online", "model": "claude-sonnet-4-20250514"}), 200


@app.route('/analyze', methods=['POST'])
def analyze_content():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No payload provided."}), 400

    content = ""
    source_url = None

    if 'url' in data:
        source_url = data['url']
        content = extract_text_from_url(source_url)
        if not content:
            return jsonify({"error": "Could not extract text from the provided URL."}), 400
    elif 'text' in data:
        content = data['text']
    else:
        return jsonify({"error": "Invalid payload format."}), 400

    if len(content) < 15:
        return jsonify({"error": "Content too short for accurate analysis."}), 400

    # Truncate to ~3000 words to stay within token limits
    truncated_content = content[:4000]

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Please analyze this content:\n\n{truncated_content}"
                }
            ]
        )

        import json
        response_text = message.content[0].text.strip()
        # Strip markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        result = json.loads(response_text)

        return jsonify({
            "verdict": result.get("verdict", "inconclusive"),
            "confidence": result.get("confidence", 0),
            "reasoning": result.get("reasoning", ""),
            "red_flags": result.get("red_flags", []),
            "text_preview": content[:150] + "..." if len(content) > 150 else content,
            "source_url": source_url
        }), 200

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse AI response: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
