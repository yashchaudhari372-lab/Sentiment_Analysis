import os
import pickle

from flask import Flask, render_template_string, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "model (1).pkl"), "rb") as f:
    model = pickle.load(f)

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
    width: 100%;
    max-width: 560px;
    border-radius: 20px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.25);
    padding: 40px;
    animation: fadeIn 0.5s ease;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
  }
  h1 {
    text-align: center;
    margin: 0 0 6px;
    font-size: 28px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  p.subtitle {
    text-align: center;
    color: #777;
    margin: 0 0 28px;
    font-size: 14px;
  }
  textarea {
    width: 100%;
    min-height: 130px;
    border-radius: 12px;
    border: 2px solid #e2e2f0;
    padding: 14px;
    font-size: 15px;
    resize: vertical;
    outline: none;
    transition: border-color 0.2s;
    font-family: inherit;
  }
  textarea:focus { border-color: #764ba2; }
  button {
    margin-top: 16px;
    width: 100%;
    padding: 14px;
    border: none;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    color: #fff;
    background: linear-gradient(135deg, #667eea, #764ba2);
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
  }
  button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(118,75,162,0.35);
  }
  .result {
    margin-top: 26px;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    animation: fadeIn 0.4s ease;
  }
  .positive {
    background: #e6f9ee;
    color: #1a9e5c;
    border: 2px solid #b5efd0;
  }
  .negative {
    background: #fdecec;
    color: #d9534f;
    border: 2px solid #f7c3c1;
  }
  .neutral {
    background: #eef1fb;
    color: #4a54a3;
    border: 2px solid #d3d8f7;
  }
  .emoji { font-size: 32px; display: block; margin-bottom: 6px; }
  .confidence {
    margin-top: 8px;
    font-size: 13px;
    font-weight: 400;
    color: #666;
  }
  .bar-track {
    margin-top: 10px;
    background: #eee;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(135deg, #667eea, #764ba2);
  }
  footer {
    text-align: center;
    margin-top: 22px;
    font-size: 12px;
    color: #aaa;
  }
</style>
</head>
<body>
  <div class="card">
    <h1>Sentiment Analyzer</h1>
    <p class="subtitle">Type a sentence and instantly see how it feels</p>
    <form method="POST">
      <textarea name="text" placeholder="e.g. I absolutely loved this product!">{{ text or "" }}</textarea>
      <button type="submit">Analyze Sentiment</button>
    </form>

    {% if sentiment %}
    <div class="result {{ css_class }}">
      <span class="emoji">{{ emoji }}</span>
      {{ sentiment }}
      {% if confidence is not none %}
      <div class="confidence">
        Confidence: {{ confidence }}%
        <div class="bar-track">
          <div class="bar-fill" style="width: {{ confidence }}%;"></div>
        </div>
      </div>
      {% endif %}
    </div>
    {% endif %}

    <footer>Powered by Flask &amp; scikit-learn</footer>
  </div>
</body>
</html>
"""


def get_emoji_and_class(label):
    label = str(label).lower()
    if "pos" in label:
        return "😊", "positive"
    if "neg" in label:
        return "😞", "negative"
    return "😐", "neutral"


@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    emoji = None
    css_class = None
    confidence = None
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            vect_text = vectorizer.transform([text])
            prediction = model.predict(vect_text)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vect_text)[0]
                confidence = round(max(proba) * 100, 2)

            sentiment = str(prediction).capitalize()
            emoji, css_class = get_emoji_and_class(prediction)

    return render_template_string(
        PAGE,
        sentiment=sentiment,
        emoji=emoji,
        css_class=css_class,
        confidence=confidence,
        text=text,
    )


if __name__ == "__main__":
    app.run(debug=True)
