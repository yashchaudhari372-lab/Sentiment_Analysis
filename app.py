from flask import Flask, render_template_string, request, jsonify
import joblib
import os
import re

app = Flask(__name__)

# ============================================================
# SENTIMENT ANALYSIS
# Model files:
#   model (1)(2).pkl       -> MultinomialNB
#   vectorizer(2).pkl      -> TF-IDF Vectorizer
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model (1).pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def clean_text(text):
    """Light cleaning while keeping useful sentiment words."""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def analyze_sentiment(text):
    text = clean_text(text)

    if not text:
        return {
            "sentiment": "Unknown",
            "confidence": 0,
            "positive": 0,
            "negative": 0,
        }

    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]

    classes = list(model.classes_)
    probability_map = {
        str(label).lower(): float(prob)
        for label, prob in zip(classes, probabilities)
    }

    sentiment = str(prediction).lower()

    # Works with the uploaded model's positive/negative classes.
    positive = probability_map.get("positive", 0.0)
    negative = probability_map.get("negative", 0.0)

    confidence = probability_map.get(sentiment, max(probabilities))

    return {
        "sentiment": sentiment.title(),
        "confidence": round(confidence * 100, 2),
        "positive": round(positive * 100, 2),
        "negative": round(negative * 100, 2),
    }


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentiment Analysis</title>

<style>
:root {
    --bg: #f5f7fb;
    --surface: rgba(255,255,255,.82);
    --surface-solid: #ffffff;
    --text: #111827;
    --muted: #6b7280;
    --border: rgba(17,24,39,.09);
    --primary: #635bff;
    --primary-2: #8b5cf6;
    --positive: #10b981;
    --negative: #ef4444;
    --shadow: 0 25px 70px rgba(15,23,42,.10);
    --soft-shadow: 0 12px 35px rgba(15,23,42,.08);
}

[data-theme="dark"] {
    --bg: #080b14;
    --surface: rgba(17,24,39,.78);
    --surface-solid: #111827;
    --text: #f8fafc;
    --muted: #94a3b8;
    --border: rgba(255,255,255,.09);
    --primary: #8b7cff;
    --primary-2: #a78bfa;
    --shadow: 0 25px 80px rgba(0,0,0,.38);
    --soft-shadow: 0 12px 35px rgba(0,0,0,.25);
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html {
    scroll-behavior: smooth;
}

body {
    min-height: 100vh;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
    background:
        radial-gradient(circle at 8% 10%, rgba(99,91,255,.16), transparent 28%),
        radial-gradient(circle at 90% 15%, rgba(139,92,246,.13), transparent 26%),
        var(--bg);
    color: var(--text);
    transition: background .35s ease, color .35s ease;
    overflow-x: hidden;
}

body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(99,91,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,91,255,.025) 1px, transparent 1px);
    background-size: 35px 35px;
    mask-image: linear-gradient(to bottom, black, transparent 80%);
}

.container {
    width: min(1180px, calc(100% - 32px));
    margin: auto;
}

/* ---------------- NAVBAR ---------------- */

.navbar {
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
    background: color-mix(in srgb, var(--surface-solid) 70%, transparent);
}

.nav-inner {
    min-height: 76px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    font-weight: 800;
    letter-spacing: -.5px;
}

.logo {
    width: 42px;
    height: 42px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    color: white;
    font-size: 19px;
    background: linear-gradient(135deg, var(--primary), var(--primary-2));
    box-shadow: 0 10px 28px rgba(99,91,255,.32);
}

.brand-text span {
    display: block;
    font-size: 15px;
}

.brand-text small {
    display: block;
    color: var(--muted);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 2px;
}

.nav-actions {
    display: flex;
    align-items: center;
    gap: 9px;
}

.icon-btn {
    width: 43px;
    height: 43px;
    border: 1px solid var(--border);
    border-radius: 13px;
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
    font-size: 17px;
    transition: .2s ease;
}

.icon-btn:hover {
    transform: translateY(-2px);
    border-color: rgba(99,91,255,.35);
    box-shadow: var(--soft-shadow);
}

/* ---------------- HERO ---------------- */

.hero {
    padding: 72px 0 34px;
    text-align: center;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 13px;
    border: 1px solid rgba(99,91,255,.18);
    border-radius: 999px;
    color: var(--primary);
    background: rgba(99,91,255,.08);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .8px;
    text-transform: uppercase;
}

.pulse {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--positive);
    box-shadow: 0 0 0 5px rgba(16,185,129,.10);
}

.hero h1 {
    margin-top: 22px;
    font-size: clamp(42px, 7vw, 76px);
    line-height: .98;
    letter-spacing: -4px;
    font-weight: 900;
}

.gradient-text {
    background: linear-gradient(120deg, var(--primary), var(--primary-2), #ec4899);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero p {
    max-width: 650px;
    margin: 22px auto 0;
    color: var(--muted);
    font-size: 16px;
    line-height: 1.7;
}

/* ---------------- DASHBOARD ---------------- */

.dashboard {
    display: grid;
    grid-template-columns: 1.35fr .85fr;
    gap: 22px;
    margin: 28px 0 65px;
}

.card {
    border: 1px solid var(--border);
    border-radius: 28px;
    background: var(--surface);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    box-shadow: var(--shadow);
}

.input-card {
    padding: 28px;
}

.card-heading {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 18px;
}

.card-heading h2 {
    font-size: 20px;
    letter-spacing: -.5px;
}

.card-heading p {
    margin-top: 5px;
    color: var(--muted);
    font-size: 13px;
}

.counter {
    color: var(--muted);
    font-size: 12px;
    white-space: nowrap;
}

textarea {
    width: 100%;
    min-height: 280px;
    resize: vertical;
    border: 1px solid var(--border);
    border-radius: 20px;
    outline: none;
    padding: 21px;
    background: var(--surface-solid);
    color: var(--text);
    font: inherit;
    line-height: 1.7;
    font-size: 15px;
    transition: .2s ease;
}

textarea:focus {
    border-color: rgba(99,91,255,.55);
    box-shadow: 0 0 0 4px rgba(99,91,255,.10);
}

.quick-prompts {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 14px 0 20px;
}

.prompt {
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--muted);
    border-radius: 999px;
    padding: 8px 12px;
    cursor: pointer;
    font-size: 12px;
    transition: .2s;
}

.prompt:hover {
    color: var(--primary);
    border-color: rgba(99,91,255,.35);
    transform: translateY(-1px);
}

.analyze-btn {
    width: 100%;
    min-height: 55px;
    border: 0;
    border-radius: 16px;
    color: white;
    cursor: pointer;
    font-size: 15px;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary), var(--primary-2));
    box-shadow: 0 15px 30px rgba(99,91,255,.26);
    transition: .2s ease;
}

.analyze-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 38px rgba(99,91,255,.34);
}

.analyze-btn:active {
    transform: translateY(0);
}

.analyze-btn.loading {
    opacity: .72;
    pointer-events: none;
}

/* ---------------- RESULT ---------------- */

.result-card {
    padding: 28px;
    position: relative;
    overflow: hidden;
}

.result-card::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: -80px;
    top: -80px;
    border-radius: 50%;
    background: rgba(99,91,255,.08);
    filter: blur(8px);
}

.result-empty {
    min-height: 470px;
    display: grid;
    place-items: center;
    text-align: center;
}

.empty-icon {
    width: 72px;
    height: 72px;
    margin: 0 auto 17px;
    display: grid;
    place-items: center;
    border-radius: 22px;
    background: rgba(99,91,255,.09);
    font-size: 30px;
}

.result-empty h3 {
    font-size: 19px;
}

.result-empty p {
    max-width: 260px;
    margin: 9px auto;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
}

.result-content {
    display: none;
}

.sentiment-pill {
    display: inline-flex;
    padding: 8px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: .8px;
    text-transform: uppercase;
}

.sentiment-positive {
    background: rgba(16,185,129,.11);
    color: var(--positive);
}

.sentiment-negative {
    background: rgba(239,68,68,.11);
    color: var(--negative);
}

.result-title {
    margin: 22px 0 8px;
    font-size: 38px;
    letter-spacing: -1.5px;
}

.confidence-label {
    color: var(--muted);
    font-size: 13px;
}

.confidence {
    margin-top: 8px;
    font-size: 43px;
    font-weight: 900;
    letter-spacing: -2px;
}

.meter {
    height: 10px;
    margin: 17px 0 25px;
    border-radius: 999px;
    background: rgba(148,163,184,.16);
    overflow: hidden;
}

.meter-fill {
    height: 100%;
    width: 0;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--primary), var(--primary-2));
    transition: width 1s cubic-bezier(.2,.8,.2,1);
}

.breakdown {
    display: grid;
    gap: 15px;
}

.metric {
    display: grid;
    gap: 8px;
}

.metric-top {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
}

.metric-top span:first-child {
    color: var(--muted);
}

.mini-bar {
    height: 8px;
    border-radius: 999px;
    background: rgba(148,163,184,.14);
    overflow: hidden;
}

.mini-fill {
    height: 100%;
    width: 0;
    border-radius: inherit;
    transition: width .8s ease;
}

.positive-fill {
    background: var(--positive);
}

.negative-fill {
    background: var(--negative);
}

.analyzed-box {
    margin-top: 25px;
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: 15px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.55;
}

/* ---------------- FEATURES ---------------- */

.features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 75px;
}

.feature {
    padding: 22px;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: var(--surface);
    box-shadow: var(--soft-shadow);
}

.feature-icon {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    border-radius: 13px;
    background: rgba(99,91,255,.09);
    margin-bottom: 16px;
}

.feature h3 {
    font-size: 15px;
    margin-bottom: 6px;
}

.feature p {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
}

/* ---------------- FOOTER ---------------- */

footer {
    padding: 25px 0 40px;
    text-align: center;
    color: var(--muted);
    font-size: 12px;
}

/* ---------------- TOAST ---------------- */

.toast {
    position: fixed;
    left: 50%;
    bottom: 25px;
    transform: translate(-50%, 30px);
    padding: 12px 17px;
    border: 1px solid var(--border);
    border-radius: 13px;
    background: var(--surface-solid);
    box-shadow: var(--shadow);
    opacity: 0;
    pointer-events: none;
    transition: .25s ease;
    z-index: 500;
    font-size: 13px;
}

.toast.show {
    opacity: 1;
    transform: translate(-50%, 0);
}

/* ---------------- RESPONSIVE ---------------- */

@media (max-width: 900px) {
    .dashboard {
        grid-template-columns: 1fr;
    }

    .features {
        grid-template-columns: 1fr;
    }

    .result-empty {
        min-height: 350px;
    }
}

@media (max-width: 560px) {
    .container {
        width: min(100% - 20px, 1180px);
    }

    .hero {
        padding-top: 50px;
    }

    .hero h1 {
        font-size: 48px;
        letter-spacing: -2.5px;
    }

    .input-card,
    .result-card {
        padding: 20px;
        border-radius: 22px;
    }

    textarea {
        min-height: 230px;
    }

    .brand-text small {
        display: none;
    }
}
</style>
</head>

<body>

<nav class="navbar">
    <div class="container nav-inner">
        <div class="brand">
            <div class="logo">✦</div>
            <div class="brand-text">
                <span>Sentiment Analysis</span>
                <small>AI Insight Dashboard</small>
            </div>
        </div>

        <div class="nav-actions">
            <button class="icon-btn" id="themeBtn" title="Toggle theme">☾</button>
        </div>
    </div>
</nav>

<main class="container">

    <section class="hero">
        <div class="badge">
            <span class="pulse"></span>
            AI-Powered Text Intelligence
        </div>

        <h1>
            Understand the <span class="gradient-text">feeling</span><br>
            behind every word.
        </h1>

        <p>
            Paste any sentence, review, feedback or message and get an instant
            sentiment prediction powered by your trained machine-learning model.
        </p>
    </section>

    <section class="dashboard">

        <div class="card input-card">
            <div class="card-heading">
                <div>
                    <h2>Analyze your text</h2>
                    <p>Enter text below and let the model evaluate its sentiment.</p>
                </div>
                <div class="counter" id="counter">0 characters</div>
            </div>

            <textarea id="textInput"
                placeholder="Example: I absolutely love this product. The quality is amazing and the service was excellent!"></textarea>

            <div class="quick-prompts">
                <button class="prompt" data-text="I absolutely love this product. The quality is amazing!">
                    ✨ Positive example
                </button>
                <button class="prompt" data-text="This experience was terrible and I am very disappointed.">
                    ⚠ Negative example
                </button>
                <button class="prompt" data-text="The product arrived today and it works as expected.">
                    ◉ Neutral-style example
                </button>
            </div>

            <button class="analyze-btn" id="analyzeBtn">
                Analyze Sentiment <span>→</span>
            </button>
        </div>

        <div class="card result-card">

            <div class="result-empty" id="emptyState">
                <div>
                    <div class="empty-icon">◎</div>
                    <h3>Your result will appear here</h3>
                    <p>
                        Submit a sentence to see the predicted sentiment,
                        confidence score and probability breakdown.
                    </p>
                </div>
            </div>

            <div class="result-content" id="resultContent">
                <span class="sentiment-pill" id="sentimentPill">Positive</span>

                <h2 class="result-title" id="sentimentTitle">Positive</h2>

                <div class="confidence-label">Model confidence</div>
                <div class="confidence" id="confidence">0%</div>

                <div class="meter">
                    <div class="meter-fill" id="confidenceFill"></div>
                </div>

                <div class="breakdown">
                    <div class="metric">
                        <div class="metric-top">
                            <span>Positive probability</span>
                            <strong id="positiveValue">0%</strong>
                        </div>
                        <div class="mini-bar">
                            <div class="mini-fill positive-fill" id="positiveFill"></div>
                        </div>
                    </div>

                    <div class="metric">
                        <div class="metric-top">
                            <span>Negative probability</span>
                            <strong id="negativeValue">0%</strong>
                        </div>
                        <div class="mini-bar">
                            <div class="mini-fill negative-fill" id="negativeFill"></div>
                        </div>
                    </div>
                </div>

                <div class="analyzed-box">
                    <strong>Analyzed text</strong><br>
                    <span id="analyzedText"></span>
                </div>
            </div>

        </div>

    </section>

    <section class="features">
        <div class="feature">
            <div class="feature-icon">⚡</div>
            <h3>Instant Prediction</h3>
            <p>Fast sentiment classification using your trained Naive Bayes model.</p>
        </div>

        <div class="feature">
            <div class="feature-icon">◈</div>
            <h3>TF-IDF Intelligence</h3>
            <p>Your uploaded TF-IDF vectorizer converts text into machine-learning features.</p>
        </div>

        <div class="feature">
            <div class="feature-icon">◐</div>
            <h3>Premium Interface</h3>
            <p>Responsive glassmorphism dashboard with light and dark presentation modes.</p>
        </div>
    </section>

</main>

<footer>
    Sentiment Analysis Dashboard · Flask + TF-IDF + Multinomial Naive Bayes
</footer>

<div class="toast" id="toast"></div>

<script>
const textInput = document.getElementById("textInput");
const counter = document.getElementById("counter");
const analyzeBtn = document.getElementById("analyzeBtn");
const themeBtn = document.getElementById("themeBtn");
const emptyState = document.getElementById("emptyState");
const resultContent = document.getElementById("resultContent");
const toast = document.getElementById("toast");

function updateCounter() {
    counter.textContent = `${textInput.value.length} characters`;
}

textInput.addEventListener("input", updateCounter);
updateCounter();

document.querySelectorAll(".prompt").forEach(button => {
    button.addEventListener("click", () => {
        textInput.value = button.dataset.text;
        updateCounter();
        textInput.focus();
    });
});

function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2200);
}

function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("sentiment-theme", theme);
    themeBtn.textContent = theme === "dark" ? "☀" : "☾";
}

const savedTheme = localStorage.getItem("sentiment-theme");
setTheme(savedTheme || "light");

themeBtn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    setTheme(current === "dark" ? "light" : "dark");
});

function animateWidth(element, value) {
    element.style.width = "0%";
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            element.style.width = `${value}%`;
        });
    });
}

async function analyze() {
    const text = textInput.value.trim();

    if (!text) {
        showToast("Please enter some text first.");
        textInput.focus();
        return;
    }

    analyzeBtn.classList.add("loading");
    analyzeBtn.innerHTML = "Analyzing <span>•••</span>";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        emptyState.style.display = "none";
        resultContent.style.display = "block";

        const isPositive = data.sentiment.toLowerCase() === "positive";

        const pill = document.getElementById("sentimentPill");
        pill.textContent = data.sentiment;
        pill.className = "sentiment-pill " +
            (isPositive ? "sentiment-positive" : "sentiment-negative");

        document.getElementById("sentimentTitle").textContent = data.sentiment;
        document.getElementById("confidence").textContent = `${data.confidence}%`;
        document.getElementById("positiveValue").textContent = `${data.positive}%`;
        document.getElementById("negativeValue").textContent = `${data.negative}%`;
        document.getElementById("analyzedText").textContent = text;

        animateWidth(document.getElementById("confidenceFill"), data.confidence);
        animateWidth(document.getElementById("positiveFill"), data.positive);
        animateWidth(document.getElementById("negativeFill"), data.negative);

        showToast(`${data.sentiment} sentiment detected.`);
    } catch (error) {
        showToast(error.message);
    } finally {
        analyzeBtn.classList.remove("loading");
        analyzeBtn.innerHTML = "Analyze Sentiment <span>→</span>";
    }
}

analyzeBtn.addEventListener("click", analyze);

textInput.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        analyze();
    }
});
</script>

</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")

        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Please enter valid text."}), 400

        result = analyze_sentiment(text)
        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "error": f"Prediction error: {str(exc)}"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
