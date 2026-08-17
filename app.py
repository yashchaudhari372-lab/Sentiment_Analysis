import os
import pickle

from flask import Flask, render_template_string, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained model + vectorizer (both .pkl files sit next to this file)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "model__1_.pkl"), "rb") as f:
    model = pickle.load(f)


# ---------------------------------------------------------------------------
# HTML template (kept inline so the whole app is a single file)
# ---------------------------------------------------------------------------
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentiment Analyzer</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }

  .card {
    background: #ffffff;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    padding: 40px;
    width: 100%;
    max-width: 560px;
    animation: fadeIn 0.5s ease-in-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
  }

  h1 {
    text-align: center;
    color: #333;
    margin-bottom: 6px;
    font-size: 28px;
  }

  p.subtitle {
    text-align: center;
    color: #888;
    margin-top: 0;
    margin-bottom: 28px;
    font-size: 14px;
  }

  textarea {
    width: 100%;
    min-height: 140px;
    border: 2px solid #e0e0e0;
    border-radius: 12px;
    padding: 14px;
    font-size: 15px;
    font-family: inherit;
    resize: vertical;
    transition: border-color 0.2s ease;
    outline: none;
  }

  textarea:focus {
    border-color: #764ba2;
  }

  button {
    width: 100%;
    margin-top: 16px;
    padding: 14px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #fff;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s ease, transform 0.1s ease;
  }

  button:hover { opacity: 0.9; }
  button:active { transform: scale(0.98); }

  .result {
    margin-top: 26px;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    animation: fadeIn 0.4s ease-in-out;
  }

  .positive {
    background: #e6f9ee;
    color: #1e7e45;
    border: 2px solid #b6f0cf;
  }

  .negative {
    background: #fdeaea;
    color: #b02a2a;
    border: 2px solid #f6c6c6;
  }

  .neutral {
    background: #eef1f7;
    color: #445;
    border: 2px solid #d7dcea;
  }

  .confidence {
    display: block;
    margin-top: 6px;
    font-size: 14px;
    font-weight: 400;
    opacity: 0.8;
  }

  .emoji { font-size: 40px; display: block; margin-bottom: 8px; }

  footer {
    text-align: center;
    margin-top: 24px;
    font-size: 12px;
    color: #aaa;
  }
</style>
</head>
<body>
  <div class="card">
    <h1>🧠 Sentiment Analyzer</h1>
    <p class="subtitle">Type a sentence and let the model tell you how it feels</p>

    <form method="POST">
      <textarea name="review_text" placeholder="e.g. This product exceeded all my expectations!">{{ review_text or '' }}</textarea>
      <button type="submit">Analyze Sentiment</button>
    </form>

    {% if sentiment %}
      <div class="result {{ css_class }}">
        <span class="emoji">{{ emoji }}</span>
        {{ sentiment }}
        {% if confidence %}
          <span class="confidence">Confidence: {{ confidence }}%</span>
        {% endif %}
      </div>
    {% endif %}

    <footer>Built with Flask &middot; TF-IDF + Naive Bayes</footer>
  </div>
</body>
</html>
"""


def analyze(text: str):
    """Run the pipeline on a single piece of text and return display info."""
    vect_text = vectorizer.transform([text])
    prediction = model.predict(vect_text)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vect_text)[0]
        confidence = round(max(proba) * 100, 2)

    label = str(prediction).lower()
    if "pos" in label:
        css_class, emoji, sentiment = "positive", "😊", "Positive"
    elif "neg" in label:
        css_class, emoji, sentiment = "negative", "😞", "Negative"
    else:
        css_class, emoji, sentiment = "neutral", "😐", str(prediction).title()

    return sentiment, css_class, emoji, confidence


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = css_class = emoji = confidence = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review_text", "").strip()
        if review_text:
            sentiment, css_class, emoji, confidence = analyze(review_text)

    return render_template_string(
        PAGE,
        review_text=review_text,
        sentiment=sentiment,
        css_class=css_class,
        emoji=emoji,
        confidence=confidence,
    )


if __name__ == "__main__":
    # Local dev server. On Render, gunicorn runs the app instead (see Procfile).
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
