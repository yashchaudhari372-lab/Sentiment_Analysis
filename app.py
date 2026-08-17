from flask import Flask, request, render_template_string, jsonify
import pickle
import os

# ============================================================
# SENTIMENT ANALYSIS - FLASK APPLICATION
# Model     : Multinomial Naive Bayes
# Vectorizer: TF-IDF
# ============================================================

app = Flask(__name__)

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
                # Transform text using your saved TF-IDF vectorizer
                transformed_text = vectorizer.transform([text])

                # Prediction
                prediction = model.predict(transformed_text)[0]

                # Confidence
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
# Useful if you want to connect another frontend later
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
# PREMIUM FRONTEND
# ============================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Sentiment Analysis | Sentiment Intelligence</title>

<!-- Google Font -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<!-- Font Awesome -->
<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">

<style>

:root {

    --bg: #070b14;
    --card: rgba(17, 24, 39, 0.72);
    --card-solid: #111827;

    --text: #f8fafc;
    --muted: #94a3b8;

    --primary: #8b5cf6;
    --primary-2: #6366f1;

    --border: rgba(255,255,255,0.09);

    --success: #22c55e;
    --danger: #ef4444;
    --warning: #f59e0b;

    --shadow:
        0 25px 80px rgba(0,0,0,0.35);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {

    font-family: "DM Sans", sans-serif;

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(139,92,246,0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(59,130,246,0.14),
            transparent 30%
        ),
        var(--bg);

    color: var(--text);

    min-height: 100vh;

    overflow-x: hidden;
}


/* =========================================================
   BACKGROUND EFFECTS
========================================================= */

.bg-orb {

    position: fixed;

    width: 350px;
    height: 350px;

    border-radius: 50%;

    filter: blur(90px);

    opacity: 0.15;

    pointer-events: none;

    z-index: -1;

    animation: floatOrb 10s ease-in-out infinite;
}

.orb-one {

    background: #8b5cf6;

    top: 5%;
    left: -100px;
}

.orb-two {

    background: #06b6d4;

    bottom: 5%;
    right: -100px;

    animation-delay: -4s;
}

@keyframes floatOrb {

    0%,100% {
        transform: translate(0,0);
    }

    50% {
        transform: translate(40px,-30px);
    }
}


/* =========================================================
   NAVBAR
========================================================= */

.navbar {

    width: 100%;

    padding: 22px 6%;

    display: flex;

    align-items: center;

    justify-content: space-between;

    border-bottom: 1px solid var(--border);

    background: rgba(7,11,20,0.65);

    backdrop-filter: blur(20px);

    position: sticky;

    top: 0;

    z-index: 100;
}

.logo {

    display: flex;

    align-items: center;

    gap: 12px;

    font-family: "Outfit";

    font-weight: 800;

    font-size: 22px;
}

.logo-icon {

    width: 42px;
    height: 42px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 13px;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--primary-2)
        );

    box-shadow:
        0 8px 30px rgba(139,92,246,0.35);
}

.logo span {

    background:
        linear-gradient(
            90deg,
            #fff,
            #c4b5fd
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.nav-actions {

    display: flex;

    align-items: center;

    gap: 10px;
}


/* =========================================================
   THEME BUTTONS
========================================================= */

.theme-selector {

    display: flex;

    gap: 6px;

    padding: 5px;

    background: rgba(255,255,255,0.05);

    border: 1px solid var(--border);

    border-radius: 12px;
}

.theme-btn {

    width: 34px;
    height: 30px;

    border: none;

    border-radius: 8px;

    cursor: pointer;

    transition: 0.25s;

    color: white;

    font-size: 12px;
}

.theme-btn:hover {
    transform: translateY(-2px);
}

.theme-purple {
    background: linear-gradient(135deg,#8b5cf6,#6366f1);
}

.theme-ocean {
    background: linear-gradient(135deg,#06b6d4,#2563eb);
}

.theme-emerald {
    background: linear-gradient(135deg,#10b981,#059669);
}

.theme-sunset {
    background: linear-gradient(135deg,#f97316,#ec4899);
}


/* =========================================================
   HERO
========================================================= */

.hero {

    width: min(1150px, 92%);

    margin: 75px auto 40px;

    text-align: center;
}

.badge {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 8px 15px;

    border: 1px solid rgba(139,92,246,0.25);

    border-radius: 100px;

    background: rgba(139,92,246,0.08);

    color: #c4b5fd;

    font-size: 13px;

    font-weight: 600;

    margin-bottom: 22px;
}

.badge i {
    color: #a78bfa;
}

.hero h1 {

    font-family: "Outfit";

    font-size: clamp(42px, 6vw, 72px);

    line-height: 1.05;

    letter-spacing: -2px;

    font-weight: 800;

    margin-bottom: 20px;
}

.gradient-text {

    background:
        linear-gradient(
            90deg,
            #a78bfa,
            #60a5fa,
            #22d3ee
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.hero p {

    max-width: 650px;

    margin: auto;

    color: var(--muted);

    font-size: 17px;

    line-height: 1.7;
}


/* =========================================================
   MAIN CARD
========================================================= */

.container {

    width: min(1000px, 92%);

    margin: 0 auto 80px;
}

.main-card {

    position: relative;

    background: var(--card);

    border: 1px solid var(--border);

    border-radius: 28px;

    padding: 32px;

    backdrop-filter: blur(25px);

    box-shadow: var(--shadow);

    overflow: hidden;
}

.main-card::before {

    content: "";

    position: absolute;

    width: 300px;
    height: 300px;

    border-radius: 50%;

    background: var(--primary);

    filter: blur(120px);

    opacity: 0.06;

    top: -160px;
    right: -100px;
}


/* =========================================================
   CARD HEADER
========================================================= */

.card-header {

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 20px;

    margin-bottom: 25px;
}

.card-title {

    display: flex;

    align-items: center;

    gap: 13px;
}

.card-title-icon {

    width: 45px;
    height: 45px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 14px;

    background:
        rgba(139,92,246,0.12);

    color: #a78bfa;
}

.card-title h2 {

    font-family: "Outfit";

    font-size: 20px;
}

.card-title p {

    color: var(--muted);

    font-size: 13px;

    margin-top: 3px;
}

.status {

    display: flex;

    align-items: center;

    gap: 7px;

    padding: 7px 12px;

    border-radius: 100px;

    background: rgba(34,197,94,0.08);

    border: 1px solid rgba(34,197,94,0.15);

    color: #86efac;

    font-size: 12px;

    font-weight: 600;
}

.status-dot {

    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #22c55e;

    box-shadow: 0 0 10px #22c55e;

    animation: pulse 1.8s infinite;
}

@keyframes pulse {

    0%,100% {
        opacity: 1;
    }

    50% {
        opacity: .4;
    }
}


/* =========================================================
   TEXT AREA
========================================================= */

.input-wrapper {

    position: relative;
}

textarea {

    width: 100%;

    min-height: 190px;

    resize: vertical;

    border: 1px solid var(--border);

    border-radius: 20px;

    background: rgba(0,0,0,0.2);

    color: var(--text);

    padding: 22px;

    font-family: "DM Sans";

    font-size: 16px;

    line-height: 1.7;

    outline: none;

    transition: 0.3s;
}

textarea::placeholder {
    color: #64748b;
}

textarea:focus {

    border-color: rgba(139,92,246,0.6);

    box-shadow:
        0 0 0 4px rgba(139,92,246,0.08);
}

.counter {

    position: absolute;

    bottom: 14px;
    right: 17px;

    color: #64748b;

    font-size: 12px;
}


/* =========================================================
   SAMPLE TEXT
========================================================= */

.samples {

    display: flex;

    flex-wrap: wrap;

    gap: 8px;

    margin-top: 15px;
}

.sample {

    border: 1px solid var(--border);

    background: rgba(255,255,255,0.03);

    color: #cbd5e1;

    border-radius: 10px;

    padding: 8px 12px;

    font-size: 12px;

    cursor: pointer;

    transition: .25s;
}

.sample:hover {

    border-color: rgba(139,92,246,.4);

    background: rgba(139,92,246,.08);

    transform: translateY(-1px);
}


/* =========================================================
   BUTTON
========================================================= */

.analyze-btn {

    width: 100%;

    margin-top: 22px;

    border: none;

    border-radius: 16px;

    padding: 17px;

    font-size: 15px;

    font-weight: 700;

    color: white;

    cursor: pointer;

    font-family: "DM Sans";

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--primary-2)
        );

    box-shadow:
        0 12px 35px rgba(99,102,241,0.25);

    transition: .3s;

    position: relative;

    overflow: hidden;
}

.analyze-btn:hover {

    transform: translateY(-3px);

    box-shadow:
        0 18px 45px rgba(99,102,241,0.38);
}

.analyze-btn:active {
    transform: scale(.99);
}

.analyze-btn i {
    margin-right: 8px;
}


/* =========================================================
   ERROR
========================================================= */

.error {

    margin-top: 20px;

    padding: 15px 17px;

    border-radius: 14px;

    background: rgba(239,68,68,0.08);

    border: 1px solid rgba(239,68,68,0.2);

    color: #fca5a5;

    font-size: 14px;
}


/* =========================================================
   RESULT
========================================================= */

.result {

    margin-top: 28px;

    border-radius: 22px;

    border: 1px solid var(--border);

    padding: 25px;

    background: rgba(255,255,255,0.025);

    animation: resultIn .5s ease;
}

@keyframes resultIn {

    from {
        opacity: 0;
        transform: translateY(15px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-header {

    display: flex;

    align-items: center;

    justify-content: space-between;

    margin-bottom: 20px;
}

.result-label {

    color: var(--muted);

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 1px;
}

.sentiment-box {

    display: flex;

    align-items: center;

    gap: 17px;
}

.sentiment-icon {

    width: 64px;
    height: 64px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 20px;

    font-size: 29px;
}

.sentiment-icon.positive {

    background: rgba(34,197,94,0.1);

    border: 1px solid rgba(34,197,94,0.2);
}

.sentiment-icon.negative {

    background: rgba(239,68,68,0.1);

    border: 1px solid rgba(239,68,68,0.2);
}

.sentiment-icon.neutral {

    background: rgba(245,158,11,0.1);

    border: 1px solid rgba(245,158,11,0.2);
}

.sentiment-name {

    font-family: "Outfit";

    font-size: 27px;

    font-weight: 800;
}

.sentiment-name.positive {
    color: #4ade80;
}

.sentiment-name.negative {
    color: #f87171;
}

.sentiment-name.neutral {
    color: #fbbf24;
}

.description {

    color: var(--muted);

    margin-top: 4px;

    font-size: 13px;
}


/* =========================================================
   CONFIDENCE
========================================================= */

.confidence {

    margin-top: 25px;
}

.confidence-top {

    display: flex;

    justify-content: space-between;

    color: var(--muted);

    font-size: 13px;

    margin-bottom: 9px;
}

.confidence-value {

    color: white;

    font-weight: 700;
}

.progress {

    height: 9px;

    background: rgba(255,255,255,.07);

    border-radius: 100px;

    overflow: hidden;
}

.progress-bar {

    height: 100%;

    border-radius: inherit;

    background:
        linear-gradient(
            90deg,
            var(--primary),
            #22d3ee
        );

    width: {{ confidence or 0 }}%;

    transition: width 1s ease;
}


/* =========================================================
   INFO CARDS
========================================================= */

.info-grid {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 14px;

    margin-top: 18px;
}

.info-card {

    padding: 18px;

    border-radius: 17px;

    border: 1px solid var(--border);

    background: rgba(255,255,255,.025);
}

.info-card i {

    color: #a78bfa;

    margin-bottom: 12px;

    font-size: 18px;
}

.info-card h3 {

    font-family: "Outfit";

    font-size: 14px;

    margin-bottom: 4px;
}

.info-card p {

    color: var(--muted);

    font-size: 12px;

    line-height: 1.5;
}


/* =========================================================
   FOOTER
========================================================= */

footer {

    text-align: center;

    padding: 30px;

    color: #64748b;

    font-size: 12px;
}

footer span {
    color: #a78bfa;
}


/* =========================================================
   RESPONSIVE
========================================================= */

@media(max-width:700px) {

    .navbar {
        padding: 18px 4%;
    }

    .hero {
        margin-top: 50px;
    }

    .hero h1 {
        letter-spacing: -1px;
    }

    .main-card {
        padding: 20px;

        border-radius: 22px;
    }

    .card-header {
        align-items: flex-start;
    }

    .status {
        display: none;
    }

    .info-grid {
        grid-template-columns: 1fr;
    }

    .theme-selector {
        gap: 3px;
    }

    .theme-btn {
        width: 30px;
    }
}


/* ============================================================
   PROFESSIONAL UI REFINEMENTS
   ============================================================ */

:root {
    --bg: #f5f7fb;
    --card: rgba(255, 255, 255, 0.94);
    --card-solid: #ffffff;
    --text: #172033;
    --muted: #64748b;
    --border: rgba(15, 23, 42, 0.10);
    --shadow: 0 20px 60px rgba(15, 23, 42, 0.10);
}

body {
    background:
        radial-gradient(circle at 8% 8%, rgba(99,102,241,0.10), transparent 28%),
        radial-gradient(circle at 92% 18%, rgba(6,182,212,0.08), transparent 28%),
        var(--bg);
    color: var(--text);
}

.navbar {
    background: rgba(255,255,255,0.88);
    border-bottom: 1px solid rgba(15,23,42,0.08);
    box-shadow: 0 4px 24px rgba(15,23,42,0.04);
}

.hero {
    margin-top: 65px;
}

.hero h1 {
    color: #172033;
    text-shadow: none;
}

.hero p {
    color: #64748b;
}

.main-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid rgba(15,23,42,0.08);
    box-shadow: 0 24px 70px rgba(15,23,42,0.10);
}

textarea {
    background: #f8fafc;
    color: #172033;
    border-color: rgba(15,23,42,0.10);
}

textarea::placeholder {
    color: #94a3b8;
}

textarea:focus {
    background: #ffffff;
}

.sample,
.info-card,
.result {
    background: #ffffff;
    border-color: rgba(15,23,42,0.08);
}

.sample {
    color: #475569;
}

.confidence-value {
    color: #172033;
}

.theme-selector {
    background: rgba(15,23,42,0.04);
    border-color: rgba(15,23,42,0.08);
}

footer {
    color: #64748b;
}

/* Improve keyboard focus visibility for accessibility. */
button:focus-visible,
textarea:focus-visible {
    outline: 3px solid rgba(99,102,241,0.22);
    outline-offset: 2px;
}

/* Cleaner mobile presentation. */
@media (max-width: 700px) {
    .hero p {
        font-size: 15px;
    }

    .hero h1 {
        font-size: clamp(38px, 12vw, 58px);
    }
}

</style>

</head>


<body>

<div class="bg-orb orb-one"></div>
<div class="bg-orb orb-two"></div>


<!-- =====================================================
     NAVBAR
===================================================== -->

<nav class="navbar">

    <div class="logo">

        <div class="logo-icon">
            <i class="fa-solid fa-brain"></i>
        </div>

        <span>Sentiment Analysis</span>

    </div>


    <div class="nav-actions">

        <div class="theme-selector">

            <button
                class="theme-btn theme-purple"
                onclick="setTheme('purple')"
                title="Purple Theme">
            </button>

            <button
                class="theme-btn theme-ocean"
                onclick="setTheme('ocean')"
                title="Ocean Theme">
            </button>

            <button
                class="theme-btn theme-emerald"
                onclick="setTheme('emerald')"
                title="Emerald Theme">
            </button>

            <button
                class="theme-btn theme-sunset"
                onclick="setTheme('sunset')"
                title="Sunset Theme">
            </button>

        </div>

    </div>

</nav>


<!-- =====================================================
     HERO
===================================================== -->

<section class="hero">

    <div class="badge">

        <i class="fa-solid fa-sparkles"></i>

        AI-POWERED SENTIMENT INTELLIGENCE

    </div>

    <h1>
        <span class="gradient-text">
            Sentiment Analysis
        </span>
    </h1>

    <p>

        Analyze text with a machine learning powered sentiment classifier.
        Get a clear positive, negative, or neutral prediction with confidence.

    </p>

</section>


<!-- =====================================================
     MAIN
===================================================== -->

<main class="container">

    <div class="main-card">


        <!-- HEADER -->

        <div class="card-header">

            <div class="card-title">

                <div class="card-title-icon">

                    <i class="fa-solid fa-message"></i>

                </div>

                <div>

                    <h2>Sentiment Analyzer</h2>

                    <p>
                        Enter your text and let the AI analyze it.
                    </p>

                </div>

            </div>


            {% if model_status %}

            <div class="status">

                <span class="status-dot"></span>

                Model Online

            </div>

            {% endif %}

        </div>


        <!-- FORM -->

        <form method="POST" id="sentimentForm">

            <div class="input-wrapper">

                <textarea
                    id="textInput"
                    name="text"
                    maxlength="5000"
                    placeholder="Type or paste your text here...

Example:
I absolutely loved this product. The quality is amazing!"
                >{{ text }}</textarea>

                <div class="counter">

                    <span id="charCount">0</span>/5000

                </div>

            </div>


            <!-- SAMPLE BUTTONS -->

            <div class="samples">

                <button
                    type="button"
                    class="sample"
                    onclick="useSample('I absolutely loved this product! The quality is amazing.')">

                    😊 Positive Example

                </button>

                <button
                    type="button"
                    class="sample"
                    onclick="useSample('This is the worst experience I have ever had.')">

                    😞 Negative Example

                </button>

                <button
                    type="button"
                    class="sample"
                    onclick="useSample('The product arrived today.')">

                    😐 Neutral Example

                </button>

            </div>


            <button class="analyze-btn" type="submit">

                <i class="fa-solid fa-wand-magic-sparkles"></i>

                Analyze Sentiment

            </button>

        </form>


        <!-- ERROR -->

        {% if error %}

        <div class="error">

            <i class="fa-solid fa-circle-exclamation"></i>

            &nbsp; {{ error }}

        </div>

        {% endif %}


        <!-- RESULT -->

        {% if result %}

        <div class="result">

            <div class="result-header">

                <span class="result-label">
                    Analysis Result
                </span>

                <i class="fa-solid fa-chart-simple"></i>

            </div>


            <div class="sentiment-box">

                <div class="sentiment-icon {{ result.class }}">

                    {{ result.emoji }}

                </div>

                <div>

                    <div class="sentiment-name {{ result.class }}">

                        {{ result.label }}

                    </div>

                    <div class="description">

                        {{ result.description }}

                    </div>

                </div>

            </div>


            {% if confidence is not none %}

            <div class="confidence">

                <div class="confidence-top">

                    <span>
                        Model Confidence
                    </span>

                    <span class="confidence-value">

                        {{ confidence }}%

                    </span>

                </div>

                <div class="progress">

                    <div class="progress-bar"></div>

                </div>

            </div>

            {% endif %}

        </div>

        {% endif %}


        <!-- INFORMATION -->

        <div class="info-grid">

            <div class="info-card">

                <i class="fa-solid fa-brain"></i>

                <h3>Machine Learning</h3>

                <p>
                    Powered by a trained Naive Bayes
                    classification model.
                </p>

            </div>


            <div class="info-card">

                <i class="fa-solid fa-chart-line"></i>

                <h3>TF-IDF Analysis</h3>

                <p>
                    Text is transformed into numerical
                    features using TF-IDF.
                </p>

            </div>


            <div class="info-card">

                <i class="fa-solid fa-bolt"></i>

                <h3>Instant Results</h3>

                <p>
                    Get sentiment predictions within
                    milliseconds.
                </p>

            </div>

        </div>


    </div>

</main>


<footer>

    Built with <span>Flask</span> •
    Machine Learning •
    Sentiment Analysis

</footer>


<script>

/* =========================================================
   CHARACTER COUNTER
========================================================= */

const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");

function updateCounter() {

    charCount.textContent =
        textInput.value.length;

}

textInput.addEventListener(
    "input",
    updateCounter
);

updateCounter();


/* =========================================================
   SAMPLE TEXT
========================================================= */

function useSample(text) {

    textInput.value = text;

    updateCounter();

    textInput.focus();

}


/* =========================================================
   PREMIUM THEMES
========================================================= */

const themes = {

    purple: {

        primary: "#8b5cf6",
        primary2: "#6366f1"

    },

    ocean: {

        primary: "#06b6d4",
        primary2: "#2563eb"

    },

    emerald: {

        primary: "#10b981",
        primary2: "#059669"

    },

    sunset: {

        primary: "#f97316",
        primary2: "#ec4899"

    }

};


function setTheme(theme) {

    const selected = themes[theme];

    document.documentElement.style.setProperty(
        "--primary",
        selected.primary
    );

    document.documentElement.style.setProperty(
        "--primary-2",
        selected.primary2
    );

    localStorage.setItem(
        "sentixTheme",
        theme
    );

}


/* =========================================================
   LOAD SAVED THEME
========================================================= */

const savedTheme =
    localStorage.getItem("sentixTheme");

if (savedTheme && themes[savedTheme]) {

    setTheme(savedTheme);

}


/* =========================================================
   FORM LOADING
========================================================= */

document
    .getElementById("sentimentForm")
    .addEventListener("submit", function() {

        const button =
            this.querySelector(".analyze-btn");

        button.innerHTML =
            '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

        button.style.pointerEvents = "none";

    });


/* =========================================================
   KEYBOARD SHORTCUT
   Ctrl + Enter = Analyze
========================================================= */

textInput.addEventListener(
    "keydown",
    function(event) {

        if (
            event.ctrlKey &&
            event.key === "Enter"
        ) {

            document
                .getElementById("sentimentForm")
                .submit();

        }

    }
);

</script>


</body>

</html>
"""


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("        SENTIX AI - SENTIMENT ANALYSIS")
    print("=" * 60)

    if MODEL_STATUS:
        print("✓ Model loaded successfully")
        print("✓ TF-IDF Vectorizer loaded successfully")
    else:
        print("✗ Model loading failed")
        print(MODEL_ERROR)

    print("=" * 60)
    print("Running at: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
