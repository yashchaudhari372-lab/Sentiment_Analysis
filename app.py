from flask import Flask, request, render_template_string, jsonify
import pickle
import os
from datetime import datetime

# ============================================================
# SENTIMENT ANALYZER
# Flask + TF-IDF + Multinomial Naive Bayes
# Professional Analytics Dashboard
# ============================================================

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model (1).pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


# ============================================================
# LOAD MODEL + VECTORIZER
# ============================================================

model = None
vectorizer = None
MODEL_STATUS = False
MODEL_ERROR = None

try:

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    MODEL_STATUS = True

except Exception as e:

    MODEL_STATUS = False
    MODEL_ERROR = str(e)


# ============================================================
# ANALYTICS MEMORY
# ============================================================

analytics = {
    "total": 0,
    "positive": 0,
    "negative": 0,
    "neutral": 0,
    "confidence_total": 0.0,
    "history": []
}


# ============================================================
# SENTIMENT DETAILS
# ============================================================

def get_sentiment_details(prediction):

    label = str(prediction).strip()
    normalized = label.lower()

    if "positive" in normalized:

        return {
            "label": "Positive",
            "emoji": "😊",
            "icon": "fa-face-smile",
            "class": "positive",
            "description": "The text expresses a positive sentiment."
        }

    elif "negative" in normalized:

        return {
            "label": "Negative",
            "emoji": "😞",
            "icon": "fa-face-frown",
            "class": "negative",
            "description": "The text expresses a negative sentiment."
        }

    elif "neutral" in normalized:

        return {
            "label": "Neutral",
            "emoji": "😐",
            "icon": "fa-face-meh",
            "class": "neutral",
            "description": "The text expresses a neutral sentiment."
        }

    return {
        "label": label.title(),
        "emoji": "🔍",
        "icon": "fa-chart-simple",
        "class": "neutral",
        "description": "The model has classified your text."
    }


# ============================================================
# PERFORM PREDICTION
# ============================================================

def analyze_text(text):

    if not MODEL_STATUS:

        raise Exception(
            "Model files could not be loaded. "
            "Please check your .pkl files."
        )

    transformed_text = vectorizer.transform([text])

    prediction = model.predict(transformed_text)[0]

    confidence = None

    try:

        probabilities = model.predict_proba(
            transformed_text
        )[0]

        confidence = round(
            float(max(probabilities)) * 100,
            2
        )

    except Exception:

        confidence = None

    details = get_sentiment_details(prediction)

    return details, confidence


# ============================================================
# UPDATE ANALYTICS
# ============================================================

def update_analytics(details, confidence, text):

    analytics["total"] += 1

    sentiment = details["label"].lower()

    if sentiment == "positive":
        analytics["positive"] += 1

    elif sentiment == "negative":
        analytics["negative"] += 1

    elif sentiment == "neutral":
        analytics["neutral"] += 1

    if confidence is not None:
        analytics["confidence_total"] += confidence

    # Keep latest 20 records
    analytics["history"].insert(
        0,
        {
            "text": text[:120],
            "sentiment": details["label"],
            "emoji": details["emoji"],
            "confidence": confidence,
            "time": datetime.now().strftime("%I:%M %p"),
            "date": datetime.now().strftime("%d %b %Y")
        }
    )

    analytics["history"] = analytics["history"][:20]


# ============================================================
# DASHBOARD DATA
# ============================================================

def get_dashboard_data():

    total = analytics["total"]

    if total > 0:
        avg_confidence = round(
            analytics["confidence_total"] / total,
            2
        )
    else:
        avg_confidence = 0

    return {
        "total": total,
        "positive": analytics["positive"],
        "negative": analytics["negative"],
        "neutral": analytics["neutral"],
        "avg_confidence": avg_confidence,
        "history": analytics["history"]
    }


# ============================================================
# MAIN ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    text = ""
    result = None
    confidence = None
    error = None

    if request.method == "POST":

        text = request.form.get(
            "text",
            ""
        ).strip()

        if not text:

            error = "Please enter some text before analyzing."

        else:

            try:

                result, confidence = analyze_text(text)

                update_analytics(
                    result,
                    confidence,
                    text
                )

            except Exception as e:

                error = str(e)

    dashboard = get_dashboard_data()

    return render_template_string(

        HTML_TEMPLATE,

        text=text,
        result=result,
        confidence=confidence,
        error=error,
        model_status=MODEL_STATUS,
        dashboard=dashboard

    )


# ============================================================
# DASHBOARD API
# ============================================================

@app.route("/api/dashboard")
def dashboard_api():

    return jsonify({
        "success": True,
        "data": get_dashboard_data()
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    if not MODEL_STATUS:

        return jsonify({
            "success": False,
            "error": "Model files could not be loaded."
        }), 500

    data = request.get_json(
        silent=True
    ) or {}

    text = str(
        data.get("text", "")
    ).strip()

    if not text:

        return jsonify({
            "success": False,
            "error": "Text is required."
        }), 400

    try:

        result, confidence = analyze_text(text)

        update_analytics(
            result,
            confidence,
            text
        )

        return jsonify({

            "success": True,

            "text": text,

            "sentiment": result["label"],

            "confidence": confidence,

            "emoji": result["emoji"]

        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# CLEAR ANALYTICS
# ============================================================

@app.route(
    "/api/clear",
    methods=["POST"]
)
def clear_analytics():

    analytics["total"] = 0
    analytics["positive"] = 0
    analytics["negative"] = 0
    analytics["neutral"] = 0
    analytics["confidence_total"] = 0
    analytics["history"] = []

    return jsonify({
        "success": True
    })


# ============================================================
# PROFESSIONAL FRONTEND
# ============================================================

HTML_TEMPLATE = r"""

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Sentiment Analyzer | Analytics Dashboard</title>


<!-- GOOGLE FONT -->

<link
href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap"
rel="stylesheet"
>


<!-- FONT AWESOME -->

<link
rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css"
>


<!-- CHART.JS -->

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>


<style>

/* ==========================================================
   THEME VARIABLES
========================================================== */

:root {

    --bg: #f5f7fb;

    --sidebar: #ffffff;

    --card: #ffffff;

    --input: #f8fafc;

    --text: #111827;

    --muted: #64748b;

    --border: #e5e7eb;

    --primary: #6366f1;

    --primary2: #8b5cf6;

    --positive: #10b981;

    --negative: #ef4444;

    --neutral: #f59e0b;

    --shadow:
        0 10px 30px rgba(15,23,42,0.07);

    --radius: 18px;
}


/* ==========================================================
   DARK THEME
========================================================== */

body.dark {

    --bg: #070b14;

    --sidebar: #0d1321;

    --card: #111827;

    --input: #0b1220;

    --text: #f8fafc;

    --muted: #94a3b8;

    --border: rgba(255,255,255,0.08);

    --shadow:
        0 15px 45px rgba(0,0,0,0.3);
}


/* ==========================================================
   OCEAN
========================================================== */

body.ocean {

    --primary: #0891b2;
    --primary2: #2563eb;
}


/* ==========================================================
   EMERALD
========================================================== */

body.emerald {

    --primary: #059669;
    --primary2: #10b981;
}


/* ==========================================================
   SUNSET
========================================================== */

body.sunset {

    --primary: #f97316;
    --primary2: #ec4899;
}


/* ==========================================================
   RESET
========================================================== */

* {

    margin: 0;
    padding: 0;

    box-sizing: border-box;
}


body {

    font-family:
        "DM Sans",
        sans-serif;

    background:
        var(--bg);

    color:
        var(--text);

    min-height: 100vh;

    transition:
        background .3s,
        color .3s;

}


/* ==========================================================
   APP LAYOUT
========================================================== */

.app {

    display: flex;

    min-height: 100vh;
}


/* ==========================================================
   SIDEBAR
========================================================== */

.sidebar {

    width: 255px;

    background:
        var(--sidebar);

    border-right:
        1px solid var(--border);

    padding: 24px 16px;

    position: fixed;

    left: 0;
    top: 0;

    bottom: 0;

    z-index: 50;

    display: flex;

    flex-direction: column;
}


/* LOGO */

.logo {

    display: flex;

    align-items: center;

    gap: 12px;

    padding: 0 10px;

    margin-bottom: 40px;
}


.logo-icon {

    width: 42px;
    height: 42px;

    border-radius: 13px;

    display: flex;

    align-items: center;

    justify-content: center;

    color: white;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--primary2)
        );

    box-shadow:
        0 8px 20px
        rgba(99,102,241,.25);
}


.logo h2 {

    font-family:
        "Outfit";

    font-size: 19px;

    font-weight: 800;
}


.logo small {

    display: block;

    color:
        var(--muted);

    font-size: 10px;

    margin-top: 2px;
}


/* NAV */

.nav-title {

    padding:
        0 12px 10px;

    font-size: 10px;

    text-transform:
        uppercase;

    letter-spacing:
        1.2px;

    color:
        var(--muted);

    font-weight: 700;
}


.nav-item {

    display: flex;

    align-items: center;

    gap: 12px;

    padding: 12px 13px;

    margin-bottom: 5px;

    border-radius: 11px;

    color:
        var(--muted);

    font-size: 13px;

    cursor: pointer;

    transition: .2s;
}


.nav-item i {

    width: 18px;

    text-align: center;
}


.nav-item:hover {

    color:
        var(--text);

    background:
        rgba(99,102,241,.07);
}


.nav-item.active {

    color:
        var(--primary);

    background:
        rgba(99,102,241,.10);

    font-weight: 700;
}


/* SIDEBAR BOTTOM */

.sidebar-bottom {

    margin-top:
        auto;
}


.model-card {

    padding: 15px;

    border-radius: 14px;

    background:
        var(--input);

    border:
        1px solid var(--border);

    margin-bottom: 15px;
}


.model-status {

    display: flex;

    align-items: center;

    gap: 8px;

    font-size: 12px;

    font-weight: 700;

    margin-bottom: 5px;
}


.green-dot {

    width: 8px;
    height: 8px;

    border-radius: 50%;

    background:
        var(--positive);

    box-shadow:
        0 0 8px
        var(--positive);
}


.model-card p {

    color:
        var(--muted);

    font-size: 11px;
}


/* ==========================================================
   MAIN CONTENT
========================================================== */

.main {

    margin-left: 255px;

    width:
        calc(100% - 255px);

    min-height: 100vh;
}


/* ==========================================================
   TOP BAR
========================================================== */

.topbar {

    height: 75px;

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding:
        0 35px;

    background:
        var(--card);

    border-bottom:
        1px solid var(--border);

    position: sticky;

    top: 0;

    z-index: 40;
}


.page-title h1 {

    font-family:
        "Outfit";

    font-size: 22px;

    font-weight: 800;
}


.page-title p {

    color:
        var(--muted);

    font-size: 12px;

    margin-top: 2px;
}


/* TOP ACTIONS */

.top-actions {

    display: flex;

    align-items: center;

    gap: 8px;
}


/* THEME BUTTONS */

.theme {

    width: 32px;
    height: 32px;

    border: none;

    border-radius: 9px;

    cursor: pointer;

    transition: .2s;
}


.theme:hover {

    transform:
        translateY(-2px);
}


.white {

    background: #ffffff;

    border:
        1px solid #cbd5e1;
}


.dark-btn {

    background:
        #111827;
}


.ocean-btn {

    background:
        linear-gradient(
            135deg,
            #06b6d4,
            #2563eb
        );
}


.green-btn {

    background:
        linear-gradient(
            135deg,
            #10b981,
            #059669
        );
}


.sunset-btn {

    background:
        linear-gradient(
            135deg,
            #f97316,
            #ec4899
        );
}


/* ==========================================================
   CONTENT
========================================================== */

.content {

    padding:
        30px 35px 50px;

    max-width:
        1600px;

    margin:
        auto;
}


/* ==========================================================
   WELCOME
========================================================== */

.welcome {

    margin-bottom:
        25px;
}


.welcome h2 {

    font-family:
        "Outfit";

    font-size: 27px;

    margin-bottom: 5px;
}


.welcome p {

    color:
        var(--muted);

    font-size: 13px;
}


/* ==========================================================
   KPI CARDS
========================================================== */

.kpi-grid {

    display:
        grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 16px;

    margin-bottom: 20px;
}


.kpi {

    background:
        var(--card);

    border:
        1px solid var(--border);

    border-radius:
        var(--radius);

    padding:
        20px;

    box-shadow:
        var(--shadow);

    transition:
        .25s;
}


.kpi:hover {

    transform:
        translateY(-3px);
}


.kpi-top {

    display: flex;

    align-items: center;

    justify-content:
        space-between;

    margin-bottom:
        15px;
}


.kpi-icon {

    width: 40px;
    height: 40px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 12px;

    background:
        rgba(99,102,241,.10);

    color:
        var(--primary);
}


.kpi-icon.green {

    color:
        var(--positive);

    background:
        rgba(16,185,129,.10);
}


.kpi-icon.red {

    color:
        var(--negative);

    background:
        rgba(239,68,68,.10);
}


.kpi-icon.yellow {

    color:
        var(--neutral);

    background:
        rgba(245,158,11,.10);
}


.kpi-label {

    color:
        var(--muted);

    font-size:
        12px;
}


.kpi-value {

    font-family:
        "Outfit";

    font-size:
        28px;

    font-weight:
        800;

    margin-top:
        4px;
}


.kpi-change {

    font-size:
        10px;

    color:
        var(--muted);

    margin-top:
        4px;
}


/* ==========================================================
   ANALYTICS GRID
========================================================== */

.analytics-grid {

    display:
        grid;

    grid-template-columns:
        1.4fr 1fr;

    gap:
        18px;

    margin-bottom:
        20px;
}


.chart-card {

    background:
        var(--card);

    border:
        1px solid var(--border);

    border-radius:
        var(--radius);

    padding:
        22px;

    box-shadow:
        var(--shadow);
}


.card-heading {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        flex-start;

    margin-bottom:
        18px;
}


.card-heading h3 {

    font-family:
        "Outfit";

    font-size:
        16px;
}


.card-heading p {

    color:
        var(--muted);

    font-size:
        11px;

    margin-top:
        3px;
}


.chart-container {

    height:
        280px;

    position:
        relative;
}


/* ==========================================================
   ANALYZER
========================================================== */

.analyzer-grid {

    display:
        grid;

    grid-template-columns:
        1.35fr .65fr;

    gap:
        18px;

    margin-bottom:
        20px;
}


.analyzer-card {

    background:
        var(--card);

    border:
        1px solid var(--border);

    border-radius:
        var(--radius);

    padding:
        24px;

    box-shadow:
        var(--shadow);
}


textarea {

    width:
        100%;

    height:
        190px;

    resize:
        vertical;

    background:
        var(--input);

    border:
        1px solid var(--border);

    border-radius:
        15px;

    padding:
        18px;

    color:
        var(--text);

    font-family:
        "DM Sans";

    font-size:
        14px;

    line-height:
        1.6;

    outline:
        none;
}


textarea:focus {

    border-color:
        var(--primary);

    box-shadow:
        0 0 0 3px
        rgba(99,102,241,.08);
}


.input-footer {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    margin-top:
        10px;
}


.samples {

    display:
        flex;

    gap:
        6px;

    flex-wrap:
        wrap;
}


.sample {

    border:
        1px solid var(--border);

    background:
        var(--input);

    color:
        var(--muted);

    border-radius:
        8px;

    padding:
        6px 9px;

    font-size:
        10px;

    cursor:
        pointer;
}


.analyze {

    width:
        100%;

    margin-top:
        18px;

    padding:
        14px;

    border:
        none;

    border-radius:
        12px;

    color:
        white;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--primary2)
        );

    font-family:
        "DM Sans";

    font-size:
        13px;

    font-weight:
        700;

    cursor:
        pointer;

    box-shadow:
        0 8px 25px
        rgba(99,102,241,.20);

    transition:
        .25s;
}


.analyze:hover {

    transform:
        translateY(-2px);
}


/* ==========================================================
   RESULT
========================================================== */

.result-panel {

    height:
        100%;

    display:
        flex;

    flex-direction:
        column;

    justify-content:
        center;

    align-items:
        center;

    text-align:
        center;
}


.result-icon {

    width:
        80px;

    height:
        80px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        24px;

    font-size:
        36px;

    margin-bottom:
        15px;
}


.result-icon.positive {

    background:
        rgba(16,185,129,.10);
}


.result-icon.negative {

    background:
        rgba(239,68,68,.10);
}


.result-icon.neutral {

    background:
        rgba(245,158,11,.10);
}


.result-name {

    font-family:
        "Outfit";

    font-size:
        27px;

    font-weight:
        800;
}


.result-name.positive {

    color:
        var(--positive);
}


.result-name.negative {

    color:
        var(--negative);
}


.result-name.neutral {

    color:
        var(--neutral);
}


.result-description {

    color:
        var(--muted);

    font-size:
        11px;

    margin-top:
        5px;

    max-width:
        230px;
}


.confidence {

    width:
        90%;

    margin-top:
        22px;
}


.confidence-top {

    display:
        flex;

    justify-content:
        space-between;

    font-size:
        11px;

    color:
        var(--muted);

    margin-bottom:
        7px;
}


.progress {

    height:
        7px;

    background:
        var(--border);

    border-radius:
        20px;

    overflow:
        hidden;
}


.progress-bar {

    height:
        100%;

    width:
        {{ confidence or 0 }}%;

    border-radius:
        inherit;

    background:
        linear-gradient(
            90deg,
            var(--primary),
            var(--primary2)
        );
}


/* ==========================================================
   HISTORY
========================================================== */

.history-card {

    background:
        var(--card);

    border:
        1px solid var(--border);

    border-radius:
        var(--radius);

    padding:
        22px;

    box-shadow:
        var(--shadow);
}


.history-header {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    margin-bottom:
        15px;
}


.clear-btn {

    border:
        1px solid var(--border);

    background:
        transparent;

    color:
        var(--muted);

    padding:
        7px 11px;

    border-radius:
        8px;

    font-size:
        10px;

    cursor:
        pointer;
}


.clear-btn:hover {

    color:
        var(--negative);

    border-color:
        var(--negative);
}


.history-row {

    display:
        grid;

    grid-template-columns:
        45px 1fr 100px 80px 90px;

    gap:
        15px;

    align-items:
        center;

    padding:
        13px 5px;

    border-bottom:
        1px solid var(--border);

    font-size:
        11px;
}


.history-row:last-child {

    border-bottom:
        none;
}


.history-icon {

    width:
        32px;

    height:
        32px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        9px;

    background:
        var(--input);

    font-size:
        15px;
}


.history-text {

    white-space:
        nowrap;

    overflow:
        hidden;

    text-overflow:
        ellipsis;

    color:
        var(--text);
}


.badge {

    display:
        inline-block;

    padding:
        5px 8px;

    border-radius:
        7px;

    font-size:
        9px;

    font-weight:
        700;
}


.badge-positive {

    color:
        var(--positive);

    background:
        rgba(16,185,129,.10);
}


.badge-negative {

    color:
        var(--negative);

    background:
        rgba(239,68,68,.10);
}


.badge-neutral {

    color:
        var(--neutral);

    background:
        rgba(245,158,11,.10);
}


.empty {

    text-align:
        center;

    padding:
        30px;

    color:
        var(--muted);

    font-size:
        12px;
}


/* ==========================================================
   FOOTER
========================================================== */

.footer {

    text-align:
        center;

    color:
        var(--muted);

    font-size:
        10px;

    padding:
        25px;
}


/* ==========================================================
   MOBILE
========================================================== */

@media(max-width:1100px) {

    .kpi-grid {

        grid-template-columns:
            repeat(2,1fr);
    }

    .analytics-grid {

        grid-template-columns:
            1fr;
    }

    .analyzer-grid {

        grid-template-columns:
            1fr;
    }
}


@media(max-width:750px) {

    .sidebar {

        width:
            65px;

        padding:
            20px 8px;
    }

    .logo {

        justify-content:
            center;

        padding:
            0;
    }

    .logo h2,
    .logo small,
    .nav-title,
    .nav-item span,
    .model-card {

        display:
            none;
    }

    .nav-item {

        justify-content:
            center;

        padding:
            12px 5px;
    }

    .main {

        margin-left:
            65px;

        width:
            calc(100% - 65px);
    }

    .topbar {

        padding:
            0 18px;
    }

    .content {

        padding:
            20px 15px;
    }

    .kpi-grid {

        grid-template-columns:
            1fr 1fr;

        gap:
            10px;
    }

    .kpi {

        padding:
            15px;
    }

    .kpi-value {

        font-size:
            22px;
    }

    .history-row {

        grid-template-columns:
            35px 1fr 80px;

    }

    .history-row .hide-mobile {

        display:
            none;
    }

}


@media(max-width:480px) {

    .page-title h1 {

        font-size:
            18px;
    }

    .theme {

        width:
            27px;

        height:
            27px;
    }

    .kpi-grid {

        grid-template-columns:
            1fr;
    }

}

</style>

</head>


<body>

<div class="app">


<!-- ======================================================
     SIDEBAR
======================================================= -->

<aside class="sidebar">


    <div class="logo">

        <div class="logo-icon">

            <i class="fa-solid fa-chart-pie"></i>

        </div>

        <div>

            <h2>Sentiment Analyzer</h2>

            <small>AI ANALYTICS</small>

        </div>

    </div>


    <div class="nav-title">
        Workspace
    </div>


    <div class="nav-item active">

        <i class="fa-solid fa-grid-2"></i>

        <span>Dashboard</span>

    </div>


    <div
        class="nav-item"
        onclick="focusAnalyzer()"
    >

        <i class="fa-solid fa-message"></i>

        <span>Analyzer</span>

    </div>


    <div
        class="nav-item"
        onclick="scrollHistory()"
    >

        <i class="fa-solid fa-clock-rotate-left"></i>

        <span>History</span>

    </div>


    <div class="nav-title" style="margin-top:25px;">
        Intelligence
    </div>


    <div class="nav-item">

        <i class="fa-solid fa-brain"></i>

        <span>Model Insights</span>

    </div>


    <div class="nav-item">

        <i class="fa-solid fa-chart-line"></i>

        <span>Analytics</span>

    </div>


    <div class="sidebar-bottom">


        <div class="model-card">

            <div class="model-status">

                <span class="green-dot"></span>

                Model Online

            </div>

            <p>
                TF-IDF + Multinomial Naive Bayes
            </p>

        </div>


        <div class="nav-item">

            <i class="fa-solid fa-gear"></i>

            <span>Settings</span>

        </div>


    </div>


</aside>


<!-- ======================================================
     MAIN
======================================================= -->

<main class="main">


<!-- TOP BAR -->

<header class="topbar">


    <div class="page-title">

        <h1>Analytics Dashboard</h1>

        <p>
            Sentiment Analyzer • Real-time model insights
        </p>

    </div>


    <div class="top-actions">


        <!-- WHITE -->

        <button
            class="theme white"
            onclick="setTheme('light')"
            title="White Theme"
        ></button>


        <!-- DARK -->

        <button
            class="theme dark-btn"
            onclick="setTheme('dark')"
            title="Dark Theme"
        ></button>


        <!-- OCEAN -->

        <button
            class="theme ocean-btn"
            onclick="setTheme('ocean')"
            title="Ocean Theme"
        ></button>


        <!-- EMERALD -->

        <button
            class="theme green-btn"
            onclick="setTheme('emerald')"
            title="Emerald Theme"
        ></button>


        <!-- SUNSET -->

        <button
            class="theme sunset-btn"
            onclick="setTheme('sunset')"
            title="Sunset Theme"
        ></button>


    </div>


</header>


<!-- CONTENT -->

<section class="content">


    <!-- WELCOME -->

    <div class="welcome">

        <h2>
            Sentiment Overview
        </h2>

        <p>
            Monitor predictions, sentiment distribution,
            confidence and recent activity.
        </p>

    </div>


    <!-- ==================================================
         KPI CARDS
    =================================================== -->

    <div class="kpi-grid">


        <!-- TOTAL -->

        <div class="kpi">

            <div class="kpi-top">

                <div class="kpi-icon">

                    <i class="fa-solid fa-chart-column"></i>

                </div>

            </div>

            <div class="kpi-label">
                Total Analyses
            </div>

            <div
                class="kpi-value"
                id="totalCount"
            >
                {{ dashboard.total }}
            </div>

            <div class="kpi-change">
                Total predictions processed
            </div>

        </div>


        <!-- POSITIVE -->

        <div class="kpi">

            <div class="kpi-top">

                <div class="kpi-icon green">

                    <i class="fa-solid fa-face-smile"></i>

                </div>

            </div>

            <div class="kpi-label">
                Positive
            </div>

            <div
                class="kpi-value"
                id="positiveCount"
            >
                {{ dashboard.positive }}
            </div>

            <div class="kpi-change">
                Positive sentiment predictions
            </div>

        </div>


        <!-- NEGATIVE -->

        <div class="kpi">

            <div class="kpi-top">

                <div class="kpi-icon red">

                    <i class="fa-solid fa-face-frown"></i>

                </div>

            </div>

            <div class="kpi-label">
                Negative
            </div>

            <div
                class="kpi-value"
                id="negativeCount"
            >
                {{ dashboard.negative }}
            </div>

            <div class="kpi-change">
                Negative sentiment predictions
            </div>

        </div>


        <!-- CONFIDENCE -->

        <div class="kpi">

            <div class="kpi-top">

                <div class="kpi-icon yellow">

                    <i class="fa-solid fa-bullseye"></i>

                </div>

            </div>

            <div class="kpi-label">
                Average Confidence
            </div>

            <div
                class="kpi-value"
                id="confidenceCount"
            >
                {{ dashboard.avg_confidence }}%
            </div>

            <div class="kpi-change">
                Average model confidence
            </div>

        </div>


    </div>


    <!-- ==================================================
         CHARTS
    =================================================== -->

    <div class="analytics-grid">


        <!-- SENTIMENT CHART -->

        <div class="chart-card">

            <div class="card-heading">

                <div>

                    <h3>
                        Sentiment Distribution
                    </h3>

                    <p>
                        Current prediction breakdown
                    </p>

                </div>

                <i class="fa-solid fa-chart-pie"></i>

            </div>


            <div class="chart-container">

                <canvas
                    id="sentimentChart"
                ></canvas>

            </div>

        </div>


        <!-- CONFIDENCE CHART -->

        <div class="chart-card">

            <div class="card-heading">

                <div>

                    <h3>
                        Sentiment Metrics
                    </h3>

                    <p>
                        Prediction volume by class
                    </p>

                </div>

                <i class="fa-solid fa-chart-simple"></i>

            </div>


            <div class="chart-container">

                <canvas
                    id="barChart"
                ></canvas>

            </div>

        </div>


    </div>


    <!-- ==================================================
         ANALYZER
    =================================================== -->

    <div
        class="analyzer-grid"
        id="analyzer"
    >


        <!-- INPUT -->

        <div class="analyzer-card">

            <div class="card-heading">

                <div>

                    <h3>
                        Analyze New Text
                    </h3>

                    <p>
                        Enter text to generate a sentiment prediction.
                    </p>

                </div>

                <i class="fa-solid fa-wand-magic-sparkles"></i>

            </div>


            <form
                method="POST"
                id="sentimentForm"
            >


                <textarea
                    name="text"
                    id="textInput"
                    maxlength="5000"
                    placeholder="Type or paste your text here..."
                >{{ text }}</textarea>


                <div class="input-footer">


                    <div class="samples">


                        <button
                            type="button"
                            class="sample"
                            onclick="useSample('I absolutely loved this product! The quality is amazing.')"
                        >
                            😊 Positive
                        </button>


                        <button
                            type="button"
                            class="sample"
                            onclick="useSample('This was the worst experience I have ever had.')"
                        >
                            😞 Negative
                        </button>


                        <button
                            type="button"
                            class="sample"
                            onclick="useSample('The package arrived today.')"
                        >
                            😐 Neutral
                        </button>


                    </div>


                    <small
                        id="counter"
                        style="color:var(--muted);"
                    >
                        0 / 5000
                    </small>


                </div>


                <button
                    class="analyze"
                    type="submit"
                >

                    <i class="fa-solid fa-wand-magic-sparkles"></i>

                    Analyze Sentiment

                </button>


            </form>


            {% if error %}

            <div
                style="
                margin-top:15px;
                color:var(--negative);
                font-size:12px;"
            >

                <i class="fa-solid fa-circle-exclamation"></i>

                {{ error }}

            </div>

            {% endif %}


        </div>


        <!-- RESULT -->

        <div class="analyzer-card">

            <div class="card-heading">

                <div>

                    <h3>
                        Latest Prediction
                    </h3>

                    <p>
                        AI model result
                    </p>

                </div>

            </div>


            <div class="result-panel">


                {% if result %}


                <div
                    class="result-icon {{ result.class }}"
                >

                    {{ result.emoji }}

                </div>


                <div
                    class="result-name {{ result.class }}"
                >

                    {{ result.label }}

                </div>


                <div class="result-description">

                    {{ result.description }}

                </div>


                {% if confidence is not none %}

                <div class="confidence">

                    <div class="confidence-top">

                        <span>
                            Confidence
                        </span>

                        <strong>
                            {{ confidence }}%
                        </strong>

                    </div>


                    <div class="progress">

                        <div
                            class="progress-bar"
                        ></div>

                    </div>

                </div>

                {% endif %}


                {% else %}


                <div
                    class="result-icon"
                    style="
                    background:var(--input);
                    color:var(--muted);"
                >

                    <i class="fa-solid fa-brain"></i>

                </div>


                <div
                    class="result-name"
                    style="font-size:20px;"
                >

                    Ready to Analyze

                </div>


                <div class="result-description">

                    Your sentiment prediction will appear here.

                </div>


                {% endif %}


            </div>


        </div>


    </div>


    <!-- ==================================================
         HISTORY
    =================================================== -->

    <div
        class="history-card"
        id="history"
    >


        <div class="history-header">


            <div>

                <h3
                    style="
                    font-family:Outfit;
                    font-size:16px;"
                >

                    Recent Analysis

                </h3>

                <p
                    style="
                    color:var(--muted);
                    font-size:11px;
                    margin-top:3px;"
                >

                    Latest sentiment predictions

                </p>

            </div>


            <button
                class="clear-btn"
                onclick="clearHistory()"
            >

                <i class="fa-solid fa-trash"></i>

                Clear History

            </button>


        </div>


        {% if dashboard.history %}


            {% for item in dashboard.history %}


            <div class="history-row">


                <div class="history-icon">

                    {{ item.emoji }}

                </div>


                <div class="history-text">

                    {{ item.text }}

                </div>


                <div>

                    {% if item.sentiment == "Positive" %}

                    <span
                        class="badge badge-positive"
                    >
                        Positive
                    </span>

                    {% elif item.sentiment == "Negative" %}

                    <span
                        class="badge badge-negative"
                    >
                        Negative
                    </span>

                    {% else %}

                    <span
                        class="badge badge-neutral"
                    >
                        Neutral
                    </span>

                    {% endif %}

                </div>


                <div
                    class="hide-mobile"
                    style="color:var(--muted);"
                >

                    {% if item.confidence %}
                        {{ item.confidence }}%
                    {% else %}
                        —
                    {% endif %}

                </div>


                <div
                    class="hide-mobile"
                    style="color:var(--muted);"
                >

                    {{ item.time }}

                </div>


            </div>


            {% endfor %}


        {% else %}


            <div class="empty">

                <i
                    class="fa-solid fa-clock-rotate-left"
                    style="
                    font-size:22px;
                    margin-bottom:8px;"
                ></i>

                <br>

                No analysis history yet.

                <br>

                Start analyzing text to populate this table.

            </div>


        {% endif %}


    </div>


</section>


<!-- FOOTER -->

<div class="footer">

    Sentiment Analyzer
    •
    Flask
    •
    TF-IDF
    •
    Multinomial Naive Bayes

</div>


</main>

</div>


<!-- ==========================================================
     JAVASCRIPT
========================================================== -->

<script>


/* ==========================================================
   DATA
========================================================== */

const dashboardData = {

    total: {{ dashboard.total }},

    positive: {{ dashboard.positive }},

    negative: {{ dashboard.negative }},

    neutral: {{ dashboard.neutral }},

    confidence: {{ dashboard.avg_confidence }}

};


/* ==========================================================
   THEME
========================================================== */

function setTheme(theme) {

    document.body.classList.remove(
        "dark",
        "ocean",
        "emerald",
        "sunset"
    );

    if (theme !== "light") {

        document.body.classList.add(
            theme
        );

    }

    localStorage.setItem(
        "sentimentTheme",
        theme
    );

}


/* LOAD SAVED THEME */

const savedTheme =
    localStorage.getItem(
        "sentimentTheme"
    );


if (savedTheme) {

    setTheme(savedTheme);

}


/* ==========================================================
   SAMPLE TEXT
========================================================== */

function useSample(text) {

    const input =
        document.getElementById(
            "textInput"
        );

    input.value = text;

    updateCounter();

    input.focus();

}


/* ==========================================================
   CHARACTER COUNTER
========================================================== */

function updateCounter() {

    const input =
        document.getElementById(
            "textInput"
        );

    const counter =
        document.getElementById(
            "counter"
        );

    counter.innerText =
        input.value.length +
        " / 5000";

}


document
    .getElementById("textInput")
    .addEventListener(
        "input",
        updateCounter
    );


updateCounter();


/* ==========================================================
   SENTIMENT DOUGHNUT CHART
========================================================== */

const sentimentCanvas =
    document.getElementById(
        "sentimentChart"
    );


const sentimentChart =
    new Chart(
        sentimentCanvas,
        {

            type: "doughnut",

            data: {

                labels: [
                    "Positive",
                    "Negative",
                    "Neutral"
                ],

                datasets: [{

                    data: [

                        dashboardData.positive,

                        dashboardData.negative,

                        dashboardData.neutral

                    ],

                    backgroundColor: [

                        "#10b981",

                        "#ef4444",

                        "#f59e0b"

                    ],

                    borderWidth: 0,

                    hoverOffset: 8

                }]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                cutout: "72%",

                plugins: {

                    legend: {

                        position: "bottom",

                        labels: {

                            padding: 18,

                            usePointStyle: true,

                            font: {
                                size: 11
                            }

                        }

                    }

                }

            }

        }
    );


/* ==========================================================
   BAR CHART
========================================================== */

const barCanvas =
    document.getElementById(
        "barChart"
    );


const barChart =
    new Chart(
        barCanvas,
        {

            type: "bar",

            data: {

                labels: [
                    "Positive",
                    "Negative",
                    "Neutral"
                ],

                datasets: [{

                    label:
                        "Predictions",

                    data: [

                        dashboardData.positive,

                        dashboardData.negative,

                        dashboardData.neutral

                    ],

                    backgroundColor: [

                        "#10b981",

                        "#ef4444",

                        "#f59e0b"

                    ],

                    borderRadius: 8,

                    borderSkipped: false

                }]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                scales: {

                    y: {

                        beginAtZero: true,

                        ticks: {
                            precision: 0
                        },

                        grid: {
                            color:
                                "rgba(148,163,184,.12)"
                        }

                    },

                    x: {

                        grid: {
                            display: false
                        }

                    }

                },

                plugins: {

                    legend: {
                        display: false
                    }

                }

            }

        }
    );


/* ==========================================================
   FORM LOADING
========================================================== */

document
    .getElementById(
        "sentimentForm"
    )
    .addEventListener(
        "submit",
        function() {

            const button =
                this.querySelector(
                    ".analyze"
                );

            button.innerHTML =
                '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

            button.disabled = true;

        }
    );


/* ==========================================================
   NAVIGATION
========================================================== */

function focusAnalyzer() {

    document
        .getElementById("analyzer")
        .scrollIntoView({
            behavior: "smooth"
        });

}


function scrollHistory() {

    document
        .getElementById("history")
        .scrollIntoView({
            behavior: "smooth"
        });

}


/* ==========================================================
   CLEAR HISTORY
========================================================== */

async function clearHistory() {

    const confirmed =
        confirm(
            "Clear all analysis history?"
        );

    if (!confirmed) {
        return;
    }


    try {

        await fetch(
            "/api/clear",
            {
                method: "POST"
            }
        );

        location.reload();

    }

    catch (error) {

        console.error(error);

    }

}


/* ==========================================================
   CTRL + ENTER
========================================================== */

document
    .getElementById("textInput")
    .addEventListener(
        "keydown",
        function(event) {

            if (
                event.ctrlKey &&
                event.key === "Enter"
            ) {

                document
                    .getElementById(
                        "sentimentForm"
                    )
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

    print("=" * 65)

    print(
        "          SENTIMENT ANALYZER"
    )

    print(
        "          ANALYTICS DASHBOARD"
    )

    print("=" * 65)

    if MODEL_STATUS:

        print(
            "✓ Model loaded successfully"
        )

        print(
            "✓ TF-IDF Vectorizer loaded successfully"
        )

    else:

        print(
            "✗ Model loading failed"
        )

        print(
            MODEL_ERROR
        )

    print("=" * 65)

    print(
        "Running at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 65)


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
