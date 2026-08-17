from flask import Flask, render_template_string, request, jsonify
import pickle
import os
import numpy as np

# ============================================================
# SENTIMENT ANALYSIS - FLASK APPLICATION
# Model: Multinomial Naive Bayes
# Vectorizer: TF-IDF
# ============================================================

app = Flask(__name__)

# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model (1).pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


# ------------------------------------------------------------
# LOAD MODEL & VECTORIZER
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
# SENTIMENT LABEL DETECTION
# ------------------------------------------------------------

def get_sentiment_label(prediction):
    """
    Converts model prediction into a readable sentiment.
    Supports numeric and text-based labels.
    """

    value = str(prediction).strip().lower()

    # Text labels
    if value in ["positive", "pos", "1", "good"]:
        return "Positive"

    if value in ["negative", "neg", "0", "bad"]:
        return "Negative"

    if value in ["neutral", "neu", "2"]:
        return "Neutral"

    # If model uses numeric classes
    try:
        number = int(float(value))

        if number == 1:
            return "Positive"

        if number == 0:
            return "Negative"

        if number == 2:
            return "Neutral"

    except:
        pass

    # Fallback
    return str(prediction).title()


# ------------------------------------------------------------
# SENTIMENT INFORMATION
# ------------------------------------------------------------

def sentiment_details(sentiment):

    if sentiment.lower() == "positive":
        return {
            "emoji": "😊",
            "icon": "fa-face-smile",
            "color": "positive",
            "message": "This text expresses a positive feeling.",
            "description": "The sentiment appears optimistic, happy, or favorable."
        }

    elif sentiment.lower() == "negative":
        return {
            "emoji": "😞",
            "icon": "fa-face-frown",
            "color": "negative",
            "message": "This text expresses a negative feeling.",
            "description": "The sentiment appears unhappy, critical, or unfavorable."
        }

    else:
        return {
            "emoji": "😐",
            "icon": "fa-face-meh",
            "color": "neutral",
            "message": "This text appears neutral.",
            "description": "The sentiment does not strongly indicate positive or negative emotion."
        }


# ------------------------------------------------------------
# CALCULATE CONFIDENCE
# ------------------------------------------------------------

def calculate_confidence(features):

    try:
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(features)[0]

            confidence = float(np.max(probabilities)) * 100

            return round(confidence, 2)

    except Exception:
        pass

    return 0.0


# ------------------------------------------------------------
# MAIN PAGE
# ------------------------------------------------------------

HTML = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Sentiment Analysis</title>

<!-- Google Font -->
<link rel="preconnect"
      href="https://fonts.googleapis.com">

<link rel="preconnect"
      href="https://fonts.gstatic.com"
      crossorigin>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap"
      rel="stylesheet">

<!-- Font Awesome -->
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">

<style>

/* =========================================================
   GLOBAL VARIABLES
========================================================= */

:root {

    --bg-main: #070b14;
    --bg-secondary: #0d1322;

    --card: rgba(255,255,255,0.07);
    --card-border: rgba(255,255,255,0.12);

    --text-main: #ffffff;
    --text-secondary: #aab3c5;

    --primary: #7c3aed;
    --secondary: #06b6d4;

    --positive: #22c55e;
    --negative: #ef4444;
    --neutral: #f59e0b;

    --shadow:
        0 25px 70px rgba(0,0,0,0.35);

    --radius: 24px;
}


/* =========================================================
   THEMES
========================================================= */

body.theme-purple {
    --primary: #7c3aed;
    --secondary: #06b6d4;
}

body.theme-ocean {
    --primary: #0ea5e9;
    --secondary: #14b8a6;
}

body.theme-sunset {
    --primary: #f97316;
    --secondary: #ec4899;
}

body.theme-emerald {
    --primary: #10b981;
    --secondary: #22c55e;
}


/* =========================================================
   RESET
========================================================= */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {

    font-family: "Inter", sans-serif;

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(124,58,237,0.20),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 20%,
            rgba(6,182,212,0.15),
            transparent 30%
        ),

        var(--bg-main);

    color: var(--text-main);

    min-height: 100vh;

    overflow-x: hidden;

    transition:
        background 0.4s ease,
        color 0.4s ease;
}


/* =========================================================
   BACKGROUND ORBS
========================================================= */

.orb {

    position: fixed;

    border-radius: 50%;

    filter: blur(80px);

    opacity: 0.25;

    pointer-events: none;

    z-index: -1;
}

.orb.one {

    width: 300px;
    height: 300px;

    background: var(--primary);

    top: 5%;
    left: -100px;

    animation: float 8s infinite ease-in-out;
}

.orb.two {

    width: 350px;
    height: 350px;

    background: var(--secondary);

    right: -120px;
    bottom: 5%;

    animation: float 10s infinite ease-in-out reverse;
}

@keyframes float {

    0%,100% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(-30px);
    }
}


/* =========================================================
   NAVBAR
========================================================= */

.navbar {

    width: 100%;

    padding: 22px 6%;

    display: flex;

    justify-content: space-between;

    align-items: center;
}

.brand {

    display: flex;

    align-items: center;

    gap: 12px;

    font-family: "Outfit";

    font-size: 21px;

    font-weight: 800;

    letter-spacing: -0.5px;
}

.brand-icon {

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
            var(--secondary)
        );

    box-shadow:
        0 10px 30px
        rgba(124,58,237,0.3);
}

.nav-actions {

    display: flex;

    align-items: center;

    gap: 10px;
}


/* =========================================================
   BUTTONS
========================================================= */

.icon-btn {

    width: 42px;
    height: 42px;

    border: 1px solid var(--card-border);

    background: var(--card);

    color: var(--text-main);

    border-radius: 13px;

    cursor: pointer;

    transition: 0.25s;

    backdrop-filter: blur(20px);
}

.icon-btn:hover {

    transform: translateY(-2px);

    border-color: var(--primary);

    background:
        rgba(255,255,255,0.12);
}


/* =========================================================
   MAIN
========================================================= */

.container {

    width: min(1100px, 92%);

    margin: auto;

    padding: 35px 0 70px;
}


/* =========================================================
   HERO
========================================================= */

.hero {

    text-align: center;

    margin-bottom: 45px;
}

.badge {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 9px 15px;

    border-radius: 100px;

    border:
        1px solid
        rgba(124,58,237,0.35);

    background:
        rgba(124,58,237,0.10);

    color: #c4b5fd;

    font-size: 13px;

    font-weight: 600;

    margin-bottom: 20px;
}

.badge-dot {

    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 12px #22c55e;
}

.hero h1 {

    font-family: "Outfit";

    font-size:
        clamp(42px, 7vw, 76px);

    line-height: 1;

    letter-spacing: -3px;

    margin-bottom: 20px;
}

.gradient-text {

    background:
        linear-gradient(
            90deg,
            var(--primary),
            var(--secondary)
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.hero p {

    max-width: 650px;

    margin: auto;

    color: var(--text-secondary);

    font-size: 16px;

    line-height: 1.7;
}


/* =========================================================
   MAIN CARD
========================================================= */

.main-card {

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.09),
            rgba(255,255,255,0.035)
        );

    border:
        1px solid var(--card-border);

    border-radius: var(--radius);

    padding: 30px;

    box-shadow: var(--shadow);

    backdrop-filter: blur(30px);

    -webkit-backdrop-filter: blur(30px);

    position: relative;

    overflow: hidden;
}

.main-card::before {

    content: "";

    position: absolute;

    width: 300px;
    height: 300px;

    background: var(--primary);

    filter: blur(120px);

    opacity: 0.08;

    top: -180px;
    right: -100px;
}


/* =========================================================
   TEXTAREA
========================================================= */

.input-header {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 14px;
}

.input-title {

    font-weight: 700;

    display: flex;

    gap: 10px;

    align-items: center;
}

.input-title i {

    color: var(--secondary);
}

.counter {

    color: var(--text-secondary);

    font-size: 12px;
}

textarea {

    width: 100%;

    min-height: 190px;

    resize: vertical;

    border-radius: 18px;

    border:
        1px solid
        var(--card-border);

    background:
        rgba(0,0,0,0.20);

    color: var(--text-main);

    outline: none;

    padding: 20px;

    font-family: inherit;

    font-size: 15px;

    line-height: 1.7;

    transition: 0.25s;
}

textarea:focus {

    border-color: var(--primary);

    box-shadow:
        0 0 0 4px
        rgba(124,58,237,0.10);
}

textarea::placeholder {

    color: #70798c;
}


/* =========================================================
   EXAMPLES
========================================================= */

.examples {

    display: flex;

    flex-wrap: wrap;

    gap: 9px;

    margin-top: 14px;
}

.example-btn {

    border:
        1px solid
        var(--card-border);

    background:
        rgba(255,255,255,0.045);

    color: var(--text-secondary);

    padding: 8px 13px;

    border-radius: 10px;

    cursor: pointer;

    font-size: 12px;

    transition: 0.2s;
}

.example-btn:hover {

    color: white;

    border-color: var(--primary);

    transform: translateY(-1px);
}


/* =========================================================
   ACTION BUTTONS
========================================================= */

.actions {

    display: flex;

    gap: 12px;

    margin-top: 20px;
}

.analyze-btn {

    flex: 1;

    border: none;

    padding: 16px 22px;

    border-radius: 15px;

    color: white;

    font-size: 15px;

    font-weight: 700;

    cursor: pointer;

    background:
        linear-gradient(
            100deg,
            var(--primary),
            var(--secondary)
        );

    box-shadow:
        0 15px 35px
        rgba(124,58,237,0.25);

    transition: 0.25s;
}

.analyze-btn:hover {

    transform: translateY(-3px);

    box-shadow:
        0 20px 40px
        rgba(124,58,237,0.35);
}

.analyze-btn:active {

    transform: scale(0.98);
}

.clear-btn {

    padding: 16px 20px;

    border-radius: 15px;

    border:
        1px solid var(--card-border);

    background:
        rgba(255,255,255,0.05);

    color: var(--text-main);

    cursor: pointer;

    font-weight: 600;

    transition: 0.25s;
}

.clear-btn:hover {

    background:
        rgba(255,255,255,0.10);
}


/* =========================================================
   LOADING
========================================================= */

.loading {

    display: none;

    text-align: center;

    padding: 30px;
}

.spinner {

    width: 38px;
    height: 38px;

    border-radius: 50%;

    border:
        3px solid
        rgba(255,255,255,0.15);

    border-top-color: var(--secondary);

    animation: spin 0.8s linear infinite;

    margin: auto auto 12px;
}

@keyframes spin {

    to {
        transform: rotate(360deg);
    }
}


/* =========================================================
   RESULT
========================================================= */

.result {

    display: none;

    margin-top: 25px;

    padding: 28px;

    border-radius: 20px;

    background:
        rgba(255,255,255,0.045);

    border:
        1px solid var(--card-border);

    animation:
        resultIn 0.5s ease;
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

.result-content {

    display: flex;

    align-items: center;

    gap: 22px;
}

.result-emoji {

    width: 80px;
    height: 80px;

    flex-shrink: 0;

    border-radius: 22px;

    display: flex;

    align-items: center;
    justify-content: center;

    font-size: 40px;

    background:
        rgba(255,255,255,0.07);
}

.result-label {

    color: var(--text-secondary);

    font-size: 13px;

    margin-bottom: 5px;
}

.result h2 {

    font-family: "Outfit";

    font-size: 30px;

    margin-bottom: 6px;
}

.result-description {

    color: var(--text-secondary);

    font-size: 13px;
}


/* SENTIMENT COLORS */

.result.positive {

    border-color:
        rgba(34,197,94,0.35);
}

.result.positive h2 {

    color: var(--positive);
}

.result.negative {

    border-color:
        rgba(239,68,68,0.35);
}

.result.negative h2 {

    color: var(--negative);
}

.result.neutral {

    border-color:
        rgba(245,158,11,0.35);
}

.result.neutral h2 {

    color: var(--neutral);
}


/* =========================================================
   CONFIDENCE
========================================================= */

.confidence {

    margin-top: 25px;
}

.confidence-header {

    display: flex;

    justify-content: space-between;

    font-size: 13px;

    margin-bottom: 9px;
}

.confidence-header span:first-child {

    color: var(--text-secondary);
}

.confidence-bar {

    width: 100%;

    height: 9px;

    background:
        rgba(255,255,255,0.08);

    border-radius: 20px;

    overflow: hidden;
}

.confidence-fill {

    width: 0%;

    height: 100%;

    border-radius: 20px;

    background:
        linear-gradient(
            90deg,
            var(--primary),
            var(--secondary)
        );

    transition:
        width 1s cubic-bezier(.2,.8,.2,1);
}


/* =========================================================
   FEATURES
========================================================= */

.features {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 15px;

    margin-top: 18px;
}

.feature {

    padding: 20px;

    border:
        1px solid var(--card-border);

    border-radius: 18px;

    background:
        rgba(255,255,255,0.035);
}

.feature-icon {

    width: 40px;
    height: 40px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 12px;

    background:
        rgba(124,58,237,0.15);

    color: #c4b5fd;

    margin-bottom: 13px;
}

.feature h3 {

    font-size: 14px;

    margin-bottom: 7px;
}

.feature p {

    color: var(--text-secondary);

    font-size: 12px;

    line-height: 1.6;
}


/* =========================================================
   THEME PANEL
========================================================= */

.theme-panel {

    position: fixed;

    right: 20px;

    top: 80px;

    width: 210px;

    padding: 18px;

    border:
        1px solid var(--card-border);

    border-radius: 18px;

    background:
        rgba(10,15,28,0.90);

    backdrop-filter: blur(25px);

    box-shadow: var(--shadow);

    z-index: 100;

    display: none;
}

.theme-panel.show {

    display: block;

    animation:
        panelIn 0.25s ease;
}

@keyframes panelIn {

    from {
        opacity: 0;
        transform: translateY(-8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.theme-panel h4 {

    font-size: 13px;

    margin-bottom: 13px;

    color: var(--text-secondary);
}

.theme-option {

    display: flex;

    align-items: center;

    gap: 10px;

    padding: 10px;

    border-radius: 10px;

    cursor: pointer;

    font-size: 13px;

    transition: 0.2s;
}

.theme-option:hover {

    background:
        rgba(255,255,255,0.07);
}

.theme-color {

    width: 20px;
    height: 20px;

    border-radius: 50%;
}

.purple {
    background:
        linear-gradient(135deg,#7c3aed,#06b6d4);
}

.ocean {
    background:
        linear-gradient(135deg,#0ea5e9,#14b8a6);
}

.sunset {
    background:
        linear-gradient(135deg,#f97316,#ec4899);
}

.emerald {
    background:
        linear-gradient(135deg,#10b981,#22c55e);
}


/* =========================================================
   FOOTER
========================================================= */

footer {

    text-align: center;

    color: #697386;

    font-size: 12px;

    padding: 30px 0;
}

footer strong {

    color: var(--text-secondary);
}


/* =========================================================
   RESPONSIVE
========================================================= */

@media(max-width: 700px) {

    .container {
        width: 94%;
    }

    .main-card {
        padding: 20px;
    }

    .features {
        grid-template-columns: 1fr;
    }

    .result-content {
        align-items: flex-start;
    }

    .result-emoji {
        width: 65px;
        height: 65px;
        font-size: 30px;
    }

    .actions {
        flex-direction: column;
    }

    .hero h1 {
        letter-spacing: -2px;
    }
}

</style>

</head>


<body class="theme-purple">

<div class="orb one"></div>
<div class="orb two"></div>


<!-- ======================================================
     NAVBAR
====================================================== -->

<nav class="navbar">

    <div class="brand">

        <div class="brand-icon">

            <i class="fa-solid fa-brain"></i>

        </div>

        Sentiment Analysis

    </div>


    <div class="nav-actions">

        <button
            class="icon-btn"
            onclick="toggleThemePanel()"
            title="Change Theme">

            <i class="fa-solid fa-palette"></i>

        </button>


        <button
            class="icon-btn"
            onclick="toggleMode()"
            title="Toggle Light/Dark">

            <i
                id="modeIcon"
                class="fa-solid fa-moon">
            </i>

        </button>

    </div>

</nav>


<!-- ======================================================
     THEME PANEL
====================================================== -->

<div
    id="themePanel"
    class="theme-panel">

    <h4>Premium Themes</h4>

    <div
        class="theme-option"
        onclick="changeTheme('theme-purple')">

        <span class="theme-color purple"></span>

        Aurora Purple

    </div>


    <div
        class="theme-option"
        onclick="changeTheme('theme-ocean')">

        <span class="theme-color ocean"></span>

        Ocean Blue

    </div>


    <div
        class="theme-option"
        onclick="changeTheme('theme-sunset')">

        <span class="theme-color sunset"></span>

        Sunset Pink

    </div>


    <div
        class="theme-option"
        onclick="changeTheme('theme-emerald')">

        <span class="theme-color emerald"></span>

        Emerald Green

    </div>

</div>


<!-- ======================================================
     MAIN CONTENT
====================================================== -->

<main class="container">


    <!-- HERO -->

    <section class="hero">

        <div class="badge">

            <span class="badge-dot"></span>

            AI-Powered Sentiment Engine

        </div>


        <h1>

            Understand the

            <span class="gradient-text">
                Emotion
            </span>

            Behind Words

        </h1>


        <p>

            Analyze text instantly using
            machine learning and discover
            whether the sentiment is positive,
            negative, or neutral.

        </p>

    </section>



    <!-- MAIN CARD -->

    <section class="main-card">


        <div class="input-header">

            <div class="input-title">

                <i class="fa-solid fa-pen-to-square"></i>

                Enter your text

            </div>


            <div
                class="counter"
                id="counter">

                0 / 1000

            </div>

        </div>


        <textarea
            id="textInput"
            maxlength="1000"
            placeholder="Write something like: I absolutely loved this product. The quality is amazing!"
        ></textarea>


        <!-- EXAMPLES -->

        <div class="examples">

            <button
                class="example-btn"
                onclick="useExample(1)">

                😊 Positive Example

            </button>


            <button
                class="example-btn"
                onclick="useExample(2)">

                😞 Negative Example

            </button>


            <button
                class="example-btn"
                onclick="useExample(3)">

                😐 Neutral Example

            </button>

        </div>


        <!-- ACTIONS -->

        <div class="actions">

            <button
                class="analyze-btn"
                onclick="analyzeSentiment()">

                <i class="fa-solid fa-wand-magic-sparkles"></i>

                &nbsp; Analyze Sentiment

            </button>


            <button
                class="clear-btn"
                onclick="clearText()">

                <i class="fa-solid fa-rotate-left"></i>

                Clear

            </button>

        </div>


        <!-- LOADING -->

        <div
            id="loading"
            class="loading">

            <div class="spinner"></div>

            <p>Analyzing your text...</p>

        </div>


        <!-- RESULT -->

        <div
            id="result"
            class="result">

            <div class="result-content">

                <div
                    id="resultEmoji"
                    class="result-emoji">

                    😊

                </div>


                <div>

                    <div class="result-label">

                        Detected Sentiment

                    </div>


                    <h2 id="resultTitle">
                        Positive
                    </h2>


                    <p
                        id="resultDescription"
                        class="result-description">

                        This text expresses a positive feeling.

                    </p>

                </div>

            </div>


            <!-- CONFIDENCE -->

            <div class="confidence">

                <div class="confidence-header">

                    <span>
                        Model Confidence
                    </span>

                    <strong id="confidenceText">
                        0%
                    </strong>

                </div>


                <div class="confidence-bar">

                    <div
                        id="confidenceFill"
                        class="confidence-fill">
                    </div>

                </div>

            </div>

        </div>

    </section>



    <!-- FEATURES -->

    <section class="features">


        <div class="feature">

            <div class="feature-icon">

                <i class="fa-solid fa-bolt"></i>

            </div>

            <h3>
                Instant Analysis
            </h3>

            <p>
                Get sentiment predictions
                within seconds using the
                trained machine learning model.
            </p>

        </div>


        <div class="feature">

            <div class="feature-icon">

                <i class="fa-solid fa-chart-simple"></i>

            </div>

            <h3>
                Confidence Score
            </h3>

            <p>
                See how confident the model
                is about its sentiment prediction.
            </p>

        </div>


        <div class="feature">

            <div class="feature-icon">

                <i class="fa-solid fa-shield-halved"></i>

            </div>

            <h3>
                ML Powered
            </h3>

            <p>
                Powered by TF-IDF feature
                extraction and Multinomial
                Naive Bayes classification.
            </p>

        </div>

    </section>


</main>


<footer>

    Built with ❤️ using
    <strong>Python • Flask • Machine Learning</strong>

</footer>



<script>

/* =========================================================
   TEXT COUNTER
========================================================= */

const textInput =
    document.getElementById("textInput");

const counter =
    document.getElementById("counter");


textInput.addEventListener("input", function() {

    counter.innerText =
        `${this.value.length} / 1000`;

});


/* =========================================================
   EXAMPLE TEXT
========================================================= */

function useExample(number) {

    if(number === 1) {

        textInput.value =
            "I absolutely loved this product! The quality is amazing and I am very happy with my purchase.";

    }

    if(number === 2) {

        textInput.value =
            "This was a terrible experience. The product was disappointing and I am very unhappy.";

    }

    if(number === 3) {

        textInput.value =
            "The product arrived today. It is available in three different sizes.";

    }

    textInput.dispatchEvent(
        new Event("input")
    );
}


/* =========================================================
   CLEAR
========================================================= */

function clearText() {

    textInput.value = "";

    textInput.dispatchEvent(
        new Event("input")
    );

    document.getElementById("result").style.display =
        "none";
}


/* =========================================================
   SENTIMENT ANALYSIS
========================================================= */

async function analyzeSentiment() {

    const text =
        textInput.value.trim();

    if(!text) {

        alert(
            "Please enter some text first."
        );

        textInput.focus();

        return;
    }


    const loading =
        document.getElementById("loading");

    const result =
        document.getElementById("result");


    loading.style.display = "block";

    result.style.display = "none";


    try {

        const response =
            await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    text: text
                })

            });


        const data =
            await response.json();


        loading.style.display = "none";


        if(!response.ok) {

            alert(
                data.error ||
                "Something went wrong."
            );

            return;
        }


        const sentiment =
            data.sentiment.toLowerCase();


        document.getElementById(
            "resultEmoji"
        ).innerText = data.emoji;


        document.getElementById(
            "resultTitle"
        ).innerText = data.sentiment;


        document.getElementById(
            "resultDescription"
        ).innerText =
            data.description;


        document.getElementById(
            "confidenceText"
        ).innerText =
            data.confidence + "%";


        const fill =
            document.getElementById(
                "confidenceFill"
            );


        fill.style.width =
            data.confidence + "%";


        result.className =
            "result " + sentiment;


        result.style.display =
            "block";


        result.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });


    }

    catch(error) {

        loading.style.display = "none";

        alert(
            "Unable to connect to the Flask server."
        );

        console.error(error);

    }

}


/* =========================================================
   ENTER KEY
========================================================= */

textInput.addEventListener(
    "keydown",
    function(event) {

        if(
            event.ctrlKey &&
            event.key === "Enter"
        ) {

            analyzeSentiment();

        }

    }
);


/* =========================================================
   THEME PANEL
========================================================= */

function toggleThemePanel() {

    document
        .getElementById("themePanel")
        .classList.toggle("show");

}


function changeTheme(theme) {

    document.body.className =
        theme;

    localStorage.setItem(
        "sentimentTheme",
        theme
    );

    document
        .getElementById("themePanel")
        .classList.remove("show");

}


/* =========================================================
   DARK / LIGHT MODE
========================================================= */

function toggleMode() {

    const body =
        document.body;

    const icon =
        document.getElementById(
            "modeIcon"
        );


    if(
        body.classList.contains(
            "light-mode"
        )
    ) {

        body.classList.remove(
            "light-mode"
        );

        icon.className =
            "fa-solid fa-moon";

        localStorage.setItem(
            "sentimentMode",
            "dark"
        );

    }

    else {

        body.classList.add(
            "light-mode"
        );

        icon.className =
            "fa-solid fa-sun";

        localStorage.setItem(
            "sentimentMode",
            "light"
        );

    }

}


/* =========================================================
   LIGHT MODE CSS INJECTION
========================================================= */

const lightStyle =
document.createElement("style");

lightStyle.innerHTML = `

body.light-mode {

    --bg-main: #f3f6fb;

    --bg-secondary: #ffffff;

    --card: rgba(255,255,255,0.70);

    --card-border:
        rgba(15,23,42,0.10);

    --text-main: #101828;

    --text-secondary: #667085;

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(124,58,237,0.12),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 20%,
            rgba(6,182,212,0.10),
            transparent 30%
        ),

        var(--bg-main);
}

body.light-mode textarea {

    background:
        rgba(255,255,255,0.75);

    color:
        #101828;
}

body.light-mode .theme-panel {

    background:
        rgba(255,255,255,0.95);

    color:
        #101828;
}

`;

document.head.appendChild(
    lightStyle
);


/* =========================================================
   LOAD SAVED SETTINGS
========================================================= */

window.addEventListener(
    "DOMContentLoaded",
    function() {

        const savedTheme =
            localStorage.getItem(
                "sentimentTheme"
            );

        if(savedTheme) {

            document.body.className =
                savedTheme;

        }


        const savedMode =
            localStorage.getItem(
                "sentimentMode"
            );

        if(savedMode === "light") {

            document.body.classList.add(
                "light-mode"
            );

            document.getElementById(
                "modeIcon"
            ).className =
                "fa-solid fa-sun";

        }

    }
);

</script>

</body>
</html>
"""


# ------------------------------------------------------------
# HOME ROUTE
# ------------------------------------------------------------

@app.route("/")
def home():

    return render_template_string(HTML)


# ------------------------------------------------------------
# PREDICTION API
# ------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if not MODEL_STATUS:

        return jsonify({
            "error":
            "Model or vectorizer could not be loaded."
        }), 500


    try:

        data = request.get_json()

        text = data.get("text", "").strip()


        if not text:

            return jsonify({
                "error":
                "Please enter some text."
            }), 400


        # ----------------------------------------------------
        # TF-IDF TRANSFORMATION
        # ----------------------------------------------------

        transformed_text =
            vectorizer.transform([text])


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction =
            model.predict(transformed_text)[0]


        sentiment =
            get_sentiment_label(prediction)


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence =
            calculate_confidence(
                transformed_text
            )


        # ----------------------------------------------------
        # DETAILS
        # ----------------------------------------------------

        details =
            sentiment_details(sentiment)


        return jsonify({

            "sentiment":
                sentiment,

            "confidence":
                confidence,

            "emoji":
                details["emoji"],

            "description":
                details["message"],

            "details":
                details["description"]

        })


    except Exception as e:

        return jsonify({

            "error":
                f"Prediction error: {str(e)}"

        }), 500


# ------------------------------------------------------------
# RUN APPLICATION
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("        SENTIMENT ANALYSIS")
    print("=" * 60)

    if MODEL_STATUS:

        print("✓ Model loaded successfully")
        print("✓ TF-IDF vectorizer loaded successfully")

    else:

        print("✗ Model loading failed")
        print(MODEL_ERROR)

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
