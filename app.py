from flask import Flask, request, render_template_string, jsonify
import pickle
import os

# ============================================================
# SENTIMENT ANALYSIS - FLASK APPLICATION
# Model     : Multinomial Naive Bayes
# Vectorizer: TF-IDF
# ============================================================

app = Flask("Sentiment Analysis")

# ------------------------------------------------------------
# FILE PATHS
# Keep app.py, model and vectorizer in the same folder
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model (1).pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    MODEL_STATUS = True

except Exception as e:
    model = None
    vectorizer = None
    MODEL_STATUS = False
    MODEL_ERROR = str(e)


# ------------------------------------------------------------
# SENTIMENT HELPER
# ------------------------------------------------------------

def get_sentiment_details(prediction, confidence=0):
    """
    Converts model prediction into UI-friendly information.
    Works with common labels such as:
    Positive, Negative, Neutral
    """

    label = str(prediction).strip()
    normalized = label.lower()

    if "positive" in normalized:
        return {
            "label": "Positive",
            "emoji": "😊",
            "icon": "fa-face-smile",
            "class": "positive",
            "description": "This text expresses a positive sentiment.",
            "color": "positive"
        }

    elif "negative" in normalized:
        return {
            "label": "Negative",
            "emoji": "😞",
            "icon": "fa-face-frown",
            "class": "negative",
            "description": "This text expresses a negative sentiment.",
            "color": "negative"
        }

    elif "neutral" in normalized:
        return {
            "label": "Neutral",
            "emoji": "😐",
            "icon": "fa-face-meh",
            "class": "neutral",
            "description": "This text expresses a neutral sentiment.",
            "color": "neutral"
        }

    # Generic fallback
    return {
        "label": label.title(),
        "emoji": "🔍",
        "icon": "fa-magnifying-glass-chart",
        "class": "neutral",
        "description": "The model has classified your text.",
        "color": "neutral"
    }


# ------------------------------------------------------------
# MAIN ROUTE
# ------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    text = ""
    result = None
    confidence = None
    error = None

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        if not text:
            error = "Please enter some text before analyzing."

        elif not MODEL_STATUS:
            error = "Model files could not be loaded. Please check the .pkl files."

        else:
            try:
                transformed_text = vectorizer.transform([text])
                prediction = model.predict(transformed_text)[0]

                try:
                    probabilities = model.predict_proba(transformed_text)[0]
                    confidence = round(float(max(probabilities)) * 100, 2)
                except Exception:
                    confidence = None

                result = get_sentiment_details(
                    prediction,
                    confidence if confidence else 0
                )

            except Exception as e:
                error = f"Prediction error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        text=text,
        result=result,
        confidence=confidence,
        error=error,
        model_status=MODEL_STATUS
    )


# ------------------------------------------------------------
# API ROUTE
# ------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():

    if not MODEL_STATUS:
        return jsonify({
            "success": False,
            "error": "Model files could not be loaded."
        }), 500

    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({
            "success": False,
            "error": "Text is required."
        }), 400

    try:
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)[0]

        confidence = None
        try:
            probabilities = model.predict_proba(transformed_text)[0]
            confidence = round(float(max(probabilities)) * 100, 2)
        except Exception:
            pass

        details = get_sentiment_details(prediction, confidence or 0)

        return jsonify({
            "success": True,
            "text": text,
            "sentiment": details["label"],
            "confidence": confidence,
            "emoji": details["emoji"]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# PROFESSIONAL FRONTEND
# ============================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentiment Analysis | AI-Powered Insights</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">
<style>
/* (Keep your existing CSS styles here — unchanged for brevity) */
</style>
</head>

<body>
<div class="bg-orb orb-one"></div>
<div class="bg-orb orb-two"></div>

<!-- Navbar -->
<div class="navbar">
    <div class="logo">
        <div class="logo-icon"><i class="fa-solid fa-chart-line"></i></div>
        <span>Sentiment Analysis</span>
    </div>
</div>

<!-- Hero -->
<div class="hero">
    <div class="badge"><i class="fa-solid fa-brain"></i> AI-Powered Insights</div>
    <h1 class="gradient-text">Sentiment Analysis</h1>
    <p>Analyze emotions in text with AI-powered sentiment detection.</p>
</div>

<!-- Main Card -->
<div class="container">
    <div class="main-card">
        <div class="card-header">
            <div class="card-title">
                <div class="card-title-icon"><i class="fa-solid fa-chart-line"></i></div>
                <h2>Sentiment Analysis Tool</h2>
                <p>Enter text below to analyze</p>
            </div>
        </div>

        <form method="POST">
            <div class="input-wrapper">
                <textarea name="text" placeholder="Type your text here...">{{ text }}</textarea>
            </div>
            <button class="analyze-btn" type="submit"><i class="fa-solid fa-magnifying-glass"></i> Analyze Sentiment</button>
        </form>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        {% if result %}
        <div class="result">
            <div class="result-header">
                <div class="result-label">Sentiment Result</div>
            </div>
            <div class="sentiment-box">
                <div class="sentiment-icon {{ result.class }}">{{ result.emoji }}</div>
                <div>
                    <div class="sentiment-name {{ result.class }}">{{ result.label }}</div>
                    <div class="description">{{ result.description }}</div>
                </div>
            </div>
            {% if confidence %}
            <div class="confidence">
                <div class="confidence-top">
                    <span>Confidence</span>
                    <span class="confidence-value">{{ confidence }}%</span>
                </div>
                <div class="progress">
                    <div class="progress-bar" style="width: {{ confidence }}%;"></div>
                </div>
            </div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</div>

<!-- Footer -->
<footer>
    <span>Sentiment Analysis</span> © 2026
</footer>

</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
