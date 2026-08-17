from flask import Flask, request, render_template_string, jsonify
import pickle
import os
from datetime import datetime

# ============================================================
# SENTIMENT ANALYSIS
# Professional Flask Analytics Application
# ============================================================

app = Flask(__name__)

# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model (1).pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


# ============================================================
# LOAD MODEL
# ============================================================

model = None
vectorizer = None
MODEL_STATUS = False
MODEL_ERROR = None

try:

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    MODEL_STATUS = True

except Exception as e:

    MODEL_STATUS = False
    MODEL_ERROR = str(e)


# ============================================================
# ANALYTICS STORAGE
# ============================================================

analytics = {

    "total": 0,

    "positive": 0,

    "negative": 0,

    "neutral": 0,

    "confidence_total": 0,

    "history": []

}


# ============================================================
# SENTIMENT INFORMATION
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

            "description":
                "The text has been classified as positive."

        }

    elif "negative" in normalized:

        return {

            "label": "Negative",

            "emoji": "😞",

            "icon": "fa-face-frown",

            "class": "negative",

            "description":
                "The text has been classified as negative."

        }

    elif "neutral" in normalized:

        return {

            "label": "Neutral",

            "emoji": "😐",

            "icon": "fa-face-meh",

            "class": "neutral",

            "description":
                "The text has been classified as neutral."

        }

    return {

        "label": label.title(),

        "emoji": "🔍",

        "icon": "fa-chart-simple",

        "class": "neutral",

        "description":
            "The sentiment has been classified by the model."

    }


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

def analyze_text(text):

    if not MODEL_STATUS:

        raise Exception(
            "Unable to load the model or vectorizer. "
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

    details = get_sentiment_details(
        prediction
    )

    return details, confidence


# ============================================================
# UPDATE ANALYTICS
# ============================================================

def update_analytics(
    details,
    confidence,
    text
):

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

    history_item = {

        "text":
            text[:150],

        "sentiment":
            details["label"],

        "emoji":
            details["emoji"],

        "confidence":
            confidence,

        "time":
            datetime.now().strftime(
                "%I:%M %p"
            ),

        "date":
            datetime.now().strftime(
                "%d %b %Y"
            )

    }

    analytics["history"].insert(
        0,
        history_item
    )

    # Keep latest 25 records

    analytics["history"] = \
        analytics["history"][:25]


# ============================================================
# DASHBOARD DATA
# ============================================================

def dashboard_data():

    total = analytics["total"]

    if total > 0:

        average_confidence = round(

            analytics["confidence_total"]
            / total,

            2

        )

    else:

        average_confidence = 0

    return {

        "total":
            total,

        "positive":
            analytics["positive"],

        "negative":
            analytics["negative"],

        "neutral":
            analytics["neutral"],

        "avg_confidence":
            average_confidence,

        "history":
            analytics["history"]

    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
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

            error = (
                "Please enter text "
                "before analyzing."
            )

        else:

            try:

                result, confidence = \
                    analyze_text(text)

                update_analytics(

                    result,

                    confidence,

                    text

                )

            except Exception as e:

                error = str(e)

    return render_template_string(

        HTML,

        text=text,

        result=result,

        confidence=confidence,

        error=error,

        model_status=MODEL_STATUS,

        dashboard=dashboard_data()

    )


# ============================================================
# DASHBOARD API
# ============================================================

@app.route(
    "/api/dashboard"
)
def dashboard_api():

    return jsonify({

        "success": True,

        "data":
            dashboard_data()

    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict_api():

    if not MODEL_STATUS:

        return jsonify({

            "success": False,

            "error":
                "Model files are not available."

        }), 500

    data = request.get_json(
        silent=True
    ) or {}

    text = str(
        data.get(
            "text",
            ""
        )
    ).strip()

    if not text:

        return jsonify({

            "success": False,

            "error":
                "Text is required."

        }), 400

    try:

        result, confidence = \
            analyze_text(text)

        update_analytics(

            result,

            confidence,

            text

        )

        return jsonify({

            "success": True,

            "text": text,

            "sentiment":
                result["label"],

            "confidence":
                confidence,

            "emoji":
                result["emoji"]

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route(
    "/api/clear",
    methods=["POST"]
)
def clear_history():

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
# PROFESSIONAL UI
# ============================================================

HTML = r"""

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1.0"
>

<title>
Sentiment Analysis
</title>


<!-- =====================================================
     FONTS
===================================================== -->

<link
href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap"
rel="stylesheet"
>


<!-- =====================================================
     FONT AWESOME
===================================================== -->

<link
rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css"
>


<!-- =====================================================
     CHART JS
===================================================== -->

<script
src="https://cdn.jsdelivr.net/npm/chart.js">
</script>


<style>

/* =========================================================
   ROOT
========================================================= */

:root {

    --background: #f6f8fc;

    --surface: #ffffff;

    --surface-2: #f8fafc;

    --text: #111827;

    --muted: #64748b;

    --border: #e5e7eb;

    --primary: #4f46e5;

    --primary-light: #eef2ff;

    --positive: #10b981;

    --negative: #ef4444;

    --neutral: #f59e0b;

    --sidebar-width: 255px;

    --radius: 16px;

    --shadow:
        0 8px 30px
        rgba(15, 23, 42, 0.06);

}


/* =========================================================
   DARK THEME
========================================================= */

body.dark {

    --background: #080c14;

    --surface: #111827;

    --surface-2: #0f172a;

    --text: #f8fafc;

    --muted: #94a3b8;

    --border:
        rgba(255,255,255,.08);

    --primary: #818cf8;

    --primary-light:
        rgba(99,102,241,.12);

    --shadow:
        0 10px 35px
        rgba(0,0,0,.25);

}


/* =========================================================
   OCEAN THEME
========================================================= */

body.ocean {

    --primary: #0284c7;

    --primary-light:
        #e0f2fe;

}


/* =========================================================
   EMERALD THEME
========================================================= */

body.emerald {

    --primary: #059669;

    --primary-light:
        #ecfdf5;

}


/* =========================================================
   SUNSET THEME
========================================================= */

body.sunset {

    --primary: #ea580c;

    --primary-light:
        #fff7ed;

}


/* =========================================================
   RESET
========================================================= */

* {

    margin: 0;

    padding: 0;

    box-sizing: border-box;

}


body {

    background:
        var(--background);

    color:
        var(--text);

    font-family:
        "DM Sans",
        sans-serif;

    min-height: 100vh;

    transition:
        .25s;

}


/* =========================================================
   APP
========================================================= */

.app {

    display: flex;

    min-height: 100vh;

}


/* =========================================================
   SIDEBAR
========================================================= */

.sidebar {

    position: fixed;

    top: 0;

    left: 0;

    bottom: 0;

    width:
        var(--sidebar-width);

    background:
        var(--surface);

    border-right:
        1px solid var(--border);

    padding:
        24px 15px;

    z-index: 100;

    display:
        flex;

    flex-direction:
        column;

}


/* =========================================================
   BRAND
========================================================= */

.brand {

    display:
        flex;

    align-items:
        center;

    gap:
        12px;

    padding:
        0 10px;

    margin-bottom:
        38px;

}


.brand-logo {

    width:
        42px;

    height:
        42px;

    border-radius:
        12px;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            #8b5cf6
        );

    color:
        white;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    font-size:
        17px;

    box-shadow:
        0 8px 20px
        rgba(79,70,229,.18);

}


.brand-name {

    font-family:
        "Outfit";

    font-weight:
        800;

    font-size:
        18px;

}


.brand-subtitle {

    color:
        var(--muted);

    font-size:
        9px;

    margin-top:
        2px;

    letter-spacing:
        1px;

    font-weight:
        700;

}


/* =========================================================
   NAV TITLE
========================================================= */

.nav-label {

    color:
        var(--muted);

    font-size:
        9px;

    font-weight:
        700;

    text-transform:
        uppercase;

    letter-spacing:
        1.2px;

    padding:
        0 12px 9px;

}


/* =========================================================
   NAV ITEM
========================================================= */

.nav-item {

    display:
        flex;

    align-items:
        center;

    gap:
        12px;

    padding:
        11px 13px;

    border-radius:
        10px;

    color:
        var(--muted);

    font-size:
        12px;

    margin-bottom:
        4px;

    cursor:
        pointer;

    transition:
        .2s;

}


.nav-item i {

    width:
        18px;

    text-align:
        center;

}


.nav-item:hover {

    background:
        var(--surface-2);

    color:
        var(--text);

}


.nav-item.active {

    background:
        var(--primary-light);

    color:
        var(--primary);

    font-weight:
        700;

}


/* =========================================================
   SIDEBAR FOOTER
========================================================= */

.sidebar-footer {

    margin-top:
        auto;

}


.status-card {

    background:
        var(--surface-2);

    border:
        1px solid var(--border);

    padding:
        14px;

    border-radius:
        13px;

    margin-bottom:
        12px;

}


.status-line {

    display:
        flex;

    align-items:
        center;

    gap:
        7px;

    font-size:
        11px;

    font-weight:
        700;

}


.status-dot {

    width:
        7px;

    height:
        7px;

    border-radius:
        50%;

    background:
        var(--positive);

    box-shadow:
        0 0 7px
        var(--positive);

}


.status-card p {

    font-size:
        9px;

    color:
        var(--muted);

    margin-top:
        5px;

}


/* =========================================================
   MAIN
========================================================= */

.main {

    margin-left:
        var(--sidebar-width);

    width:
        calc(
            100% -
            var(--sidebar-width)
        );

}


/* =========================================================
   TOPBAR
========================================================= */

.topbar {

    height:
        72px;

    background:
        var(--surface);

    border-bottom:
        1px solid var(--border);

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    padding:
        0 32px;

    position:
        sticky;

    top:
        0;

    z-index:
        50;

}


.top-title h1 {

    font-family:
        "Outfit";

    font-size:
        21px;

    font-weight:
        800;

}


.top-title p {

    color:
        var(--muted);

    font-size:
        11px;

    margin-top:
        2px;

}


.theme-buttons {

    display:
        flex;

    gap:
        7px;

}


.theme-button {

    width:
        28px;

    height:
        28px;

    border:
        none;

    border-radius:
        8px;

    cursor:
        pointer;

    transition:
        .2s;

}


.theme-button:hover {

    transform:
        translateY(-2px);

}


.theme-light {

    background:
        #ffffff;

    border:
        1px solid #cbd5e1;

}


.theme-dark {

    background:
        #111827;

}


.theme-ocean {

    background:
        linear-gradient(
            135deg,
            #06b6d4,
            #2563eb
        );

}


.theme-emerald {

    background:
        linear-gradient(
            135deg,
            #10b981,
            #059669
        );

}


.theme-sunset {

    background:
        linear-gradient(
            135deg,
            #f97316,
            #ec4899
        );

}


/* =========================================================
   CONTENT
========================================================= */

.content {

    max-width:
        1500px;

    margin:
        auto;

    padding:
        30px 32px 45px;

}


/* =========================================================
   PAGE HEADER
========================================================= */

.page-header {

    margin-bottom:
        22px;

}


.page-header h2 {

    font-family:
        "Outfit";

    font-size:
        26px;

    font-weight:
        800;

}


.page-header p {

    color:
        var(--muted);

    font-size:
        12px;

    margin-top:
        4px;

}


/* =========================================================
   KPI
========================================================= */

.kpi-grid {

    display:
        grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap:
        15px;

    margin-bottom:
        18px;

}


.kpi {

    background:
        var(--surface);

    border:
        1px solid var(--border);

    border-radius:
        var(--radius);

    padding:
        19px;

    box-shadow:
        var(--shadow);

    transition:
        .2s;

}


.kpi:hover {

    transform:
        translateY(-2px);

}


.kpi-head {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

}


.kpi-icon {

    width:
        38px;

    height:
        38px;

    border-radius:
        11px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    background:
        var(--primary-light);

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


.kpi-icon.orange {

    color:
        var(--neutral);

    background:
        rgba(245,158,11,.10);

}


.kpi-label {

    color:
        var(--muted);

    font-size:
        11px;

    margin-top:
        15px;

}


.kpi-value {

    font-family:
        "Outfit";

    font-size:
        27px;

    font-weight:
        800;

    margin-top:
        2px;

}


.kpi-description {

    color:
        var(--muted);

    font-size:
        9px;

    margin-top:
        2px;

}


/* =========================================================
   CHART GRID
========================================================= */

.chart-grid {

    display:
        grid;

    grid-template-columns:
        1.4fr 1fr;

    gap:
        16px;

    margin-bottom:
        18px;

}


.chart-card {

    background:
        var(--surface);

    border:
        1px solid var(--border);

    border-radius:
        var(--radius);

    padding:
        20px;

    box-shadow:
        var(--shadow);

}


.card-title {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        flex-start;

    margin-bottom:
        15px;

}


.card-title h3 {

    font-family:
        "Outfit";

    font-size:
        15px;

}


.card-title p {

    color:
        var(--muted);

    font-size:
        10px;

    margin-top:
        3px;

}


.chart-area {

    height:
        260px;

}


/* =========================================================
   ANALYZER GRID
========================================================= */

.analyzer-grid {

    display:
        grid;

    grid-template-columns:
        1.25fr .75fr;

    gap:
        16px;

    margin-bottom:
        18px;

}


.analyzer-card {

    background:
        var(--surface);

    border:
        1px solid var(--border);

    border-radius:
        var(--radius);

    padding:
        22px;

    box-shadow:
        var(--shadow);

}


/* =========================================================
   TEXTAREA
========================================================= */

textarea {

    width:
        100%;

    height:
        180px;

    resize:
        vertical;

    border:
        1px solid var(--border);

    background:
        var(--surface-2);

    color:
        var(--text);

    border-radius:
        12px;

    padding:
        15px;

    font-family:
        "DM Sans";

    font-size:
        13px;

    line-height:
        1.6;

    outline:
        none;

    transition:
        .2s;

}


textarea:focus {

    border-color:
        var(--primary);

    box-shadow:
        0 0 0 3px
        var(--primary-light);

}


/* =========================================================
   TEXTAREA FOOTER
========================================================= */

.input-footer {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    margin-top:
        8px;

}


.samples {

    display:
        flex;

    gap:
        5px;

    flex-wrap:
        wrap;

}


.sample {

    border:
        1px solid var(--border);

    background:
        var(--surface-2);

    color:
        var(--muted);

    border-radius:
        7px;

    padding:
        6px 9px;

    font-size:
        9px;

    cursor:
        pointer;

}


.sample:hover {

    color:
        var(--primary);

    border-color:
        var(--primary);

}


/* =========================================================
   ANALYZE BUTTON
========================================================= */

.analyze-button {

    width:
        100%;

    border:
        none;

    border-radius:
        11px;

    padding:
        13px;

    margin-top:
        16px;

    color:
        white;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            #7c3aed
        );

    font-family:
        "DM Sans";

    font-size:
        12px;

    font-weight:
        700;

    cursor:
        pointer;

    box-shadow:
        0 8px 22px
        rgba(79,70,229,.18);

    transition:
        .2s;

}


.analyze-button:hover {

    transform:
        translateY(-2px);

}


/* =========================================================
   RESULT
========================================================= */

.result {

    height:
        100%;

    min-height:
        280px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    flex-direction:
        column;

    text-align:
        center;

}


.result-symbol {

    width:
        78px;

    height:
        78px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        22px;

    font-size:
        35px;

    margin-bottom:
        12px;

}


.result-symbol.positive {

    background:
        rgba(16,185,129,.10);

}


.result-symbol.negative {

    background:
        rgba(239,68,68,.10);

}


.result-symbol.neutral {

    background:
        rgba(245,158,11,.10);

}


.result-title {

    font-family:
        "Outfit";

    font-size:
        26px;

    font-weight:
        800;

}


.result-title.positive {

    color:
        var(--positive);

}


.result-title.negative {

    color:
        var(--negative);

}


.result-title.neutral {

    color:
        var(--neutral);

}


.result-description {

    color:
        var(--muted);

    font-size:
        10px;

    margin-top:
        4px;

}


.confidence-box {

    width:
        85%;

    margin-top:
        20px;

}


.confidence-header {

    display:
        flex;

    justify-content:
        space-between;

    color:
        var(--muted);

    font-size:
        10px;

    margin-bottom:
        6px;

}


.progress {

    height:
        7px;

    border-radius:
        10px;

    overflow:
        hidden;

    background:
        var(--border);

}


.progress-value {

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
            #8b5cf6
        );

}


/* =========================================================
   HISTORY
========================================================= */

.history-card {

    background:
        var(--surface);

    border:
        1px solid var(--border);

    border-radius:
        var(--radius);

    padding:
        20px;

    box-shadow:
        var(--shadow);

}


.history-header {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    margin-bottom:
        12px;

}


.history-header h3 {

    font-family:
        "Outfit";

    font-size:
        15px;

}


.history-header p {

    color:
        var(--muted);

    font-size:
        10px;

    margin-top:
        2px;

}


.clear-button {

    background:
        transparent;

    border:
        1px solid var(--border);

    color:
        var(--muted);

    border-radius:
        7px;

    padding:
        6px 9px;

    font-size:
        9px;

    cursor:
        pointer;

}


.clear-button:hover {

    color:
        var(--negative);

    border-color:
        var(--negative);

}


/* =========================================================
   HISTORY ROW
========================================================= */

.history-row {

    display:
        grid;

    grid-template-columns:
        40px 1fr 100px 80px 90px;

    gap:
        12px;

    align-items:
        center;

    border-bottom:
        1px solid var(--border);

    padding:
        11px 3px;

    font-size:
        10px;

}


.history-row:last-child {

    border-bottom:
        none;

}


.history-icon {

    width:
        30px;

    height:
        30px;

    border-radius:
        8px;

    background:
        var(--surface-2);

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

}


.history-text {

    white-space:
        nowrap;

    overflow:
        hidden;

    text-overflow:
        ellipsis;

}


.badge {

    display:
        inline-block;

    padding:
        4px 7px;

    border-radius:
        6px;

    font-size:
        8px;

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


/* =========================================================
   ERROR
========================================================= */

.error-message {

    margin-top:
        10px;

    padding:
        10px;

    border-radius:
        8px;

    background:
        rgba(239,68,68,.08);

    color:
        var(--negative);

    font-size:
        10px;

}


/* =========================================================
   FOOTER
========================================================= */

.footer {

    text-align:
        center;

    color:
        var(--muted);

    font-size:
        9px;

    padding:
        25px;

}


/* =========================================================
   RESPONSIVE
========================================================= */

@media(max-width:1100px) {

    .kpi-grid {

        grid-template-columns:
            repeat(2,1fr);

    }

    .chart-grid {

        grid-template-columns:
            1fr;

    }

    .analyzer-grid {

        grid-template-columns:
            1fr;

    }

}


@media(max-width:720px) {

    .sidebar {

        width:
            65px;

        padding:
            20px 7px;

    }

    .brand-name,

    .brand-subtitle,

    .nav-label,

    .nav-item span,

    .status-card {

        display:
            none;

    }

    .brand {

        justify-content:
            center;

    }

    .nav-item {

        justify-content:
            center;

    }

    .main {

        margin-left:
            65px;

        width:
            calc(
                100% - 65px
            );

    }

    .content {

        padding:
            20px 15px;

    }

    .topbar {

        padding:
            0 15px;

    }

    .history-row {

        grid-template-columns:
            35px 1fr 80px;

    }

    .history-hide {

        display:
            none;

    }

}


@media(max-width:450px) {

    .kpi-grid {

        grid-template-columns:
            1fr;

    }

    .top-title h1 {

        font-size:
            17px;

    }

    .theme-button {

        width:
            23px;

        height:
            23px;

    }

}

</style>

</head>


<body>


<div class="app">


<!-- =====================================================
     SIDEBAR
===================================================== -->

<aside class="sidebar">


    <div class="brand">

        <div class="brand-logo">

            <i class="fa-solid fa-chart-line"></i>

        </div>

        <div>

            <div class="brand-name">

                Sentiment Analysis

            </div>

            <div class="brand-subtitle">

                ANALYTICS PLATFORM

            </div>

        </div>

    </div>


    <div class="nav-label">

        Workspace

    </div>


    <div
        class="nav-item active"
        onclick="goTo('dashboard')"
    >

        <i class="fa-solid fa-table-columns"></i>

        <span>
            Dashboard
        </span>

    </div>


    <div
        class="nav-item"
        onclick="goTo('analyzer')"
    >

        <i class="fa-solid fa-magnifying-glass-chart"></i>

        <span>
            Sentiment Analyzer
        </span>

    </div>


    <div
        class="nav-item"
        onclick="goTo('history')"
    >

        <i class="fa-solid fa-clock-rotate-left"></i>

        <span>
            Analysis History
        </span>

    </div>


    <div
        class="nav-label"
        style="margin-top:25px;"
    >

        Insights

    </div>


    <div class="nav-item">

        <i class="fa-solid fa-chart-pie"></i>

        <span>
            Sentiment Metrics
        </span>

    </div>


    <div class="nav-item">

        <i class="fa-solid fa-brain"></i>

        <span>
            Model Information
        </span>

    </div>


    <div class="sidebar-footer">


        <div class="status-card">

            <div class="status-line">

                <span class="status-dot"></span>

                Model Online

            </div>

            <p>

                TF-IDF + Multinomial Naive Bayes

            </p>

        </div>


        <div class="nav-item">

            <i class="fa-solid fa-gear"></i>

            <span>
                Settings
            </span>

        </div>


    </div>


</aside>


<!-- =====================================================
     MAIN
===================================================== -->

<main class="main">


<!-- =====================================================
     TOPBAR
===================================================== -->

<header class="topbar">


    <div class="top-title">

        <h1>
            Sentiment Analysis
        </h1>

        <p>
            Professional sentiment analytics dashboard
        </p>

    </div>


    <div class="theme-buttons">


        <button
            class="theme-button theme-light"
            onclick="setTheme('light')"
            title="Light"
        ></button>


        <button
            class="theme-button theme-dark"
            onclick="setTheme('dark')"
            title="Dark"
        ></button>


        <button
            class="theme-button theme-ocean"
            onclick="setTheme('ocean')"
            title="Ocean"
        ></button>


        <button
            class="theme-button theme-emerald"
            onclick="setTheme('emerald')"
            title="Emerald"
        ></button>


        <button
            class="theme-button theme-sunset"
            onclick="setTheme('sunset')"
            title="Sunset"
        ></button>


    </div>


</header>


<!-- =====================================================
     CONTENT
===================================================== -->

<section
    class="content"
    id="dashboard"
>


    <div class="page-header">

        <h2>
            Analytics Overview
        </h2>

        <p>
            Monitor sentiment predictions and model confidence.
        </p>

    </div>


    <!-- =================================================
         KPI CARDS
    ================================================== -->

    <div class="kpi-grid">


        <div class="kpi">

            <div class="kpi-head">

                <div class="kpi-icon">

                    <i class="fa-solid fa-chart-column"></i>

                </div>

            </div>

            <div class="kpi-label">

                Total Analyses

            </div>

            <div
                class="kpi-value"
                id="total"
            >

                {{ dashboard.total }}

            </div>

            <div class="kpi-description">

                Total predictions processed

            </div>

        </div>


        <div class="kpi">

            <div class="kpi-head">

                <div class="kpi-icon green">

                    <i class="fa-solid fa-face-smile"></i>

                </div>

            </div>

            <div class="kpi-label">

                Positive

            </div>

            <div
                class="kpi-value"
                id="positive"
            >

                {{ dashboard.positive }}

            </div>

            <div class="kpi-description">

                Positive predictions

            </div>

        </div>


        <div class="kpi">

            <div class="kpi-head">

                <div class="kpi-icon red">

                    <i class="fa-solid fa-face-frown"></i>

                </div>

            </div>

            <div class="kpi-label">

                Negative

            </div>

            <div
                class="kpi-value"
                id="negative"
            >

                {{ dashboard.negative }}

            </div>

            <div class="kpi-description">

                Negative predictions

            </div>

        </div>


        <div class="kpi">

            <div class="kpi-head">

                <div class="kpi-icon orange">

                    <i class="fa-solid fa-bullseye"></i>

                </div>

            </div>

            <div class="kpi-label">

                Average Confidence

            </div>

            <div
                class="kpi-value"
                id="confidence"
            >

                {{ dashboard.avg_confidence }}%

            </div>

            <div class="kpi-description">

                Average model confidence

            </div>

        </div>


    </div>


    <!-- =================================================
         CHARTS
    ================================================== -->

    <div class="chart-grid">


        <div class="chart-card">


            <div class="card-title">

                <div>

                    <h3>
                        Sentiment Distribution
                    </h3>

                    <p>
                        Prediction distribution by sentiment
                    </p>

                </div>

                <i class="fa-solid fa-chart-pie"></i>

            </div>


            <div class="chart-area">

                <canvas
                    id="donutChart"
                ></canvas>

            </div>


        </div>


        <div class="chart-card">


            <div class="card-title">

                <div>

                    <h3>
                        Prediction Volume
                    </h3>

                    <p>
                        Number of predictions by category
                    </p>

                </div>

                <i class="fa-solid fa-chart-column"></i>

            </div>


            <div class="chart-area">

                <canvas
                    id="barChart"
                ></canvas>

            </div>


        </div>


    </div>


    <!-- =================================================
         ANALYZER
    ================================================== -->

    <div
        class="analyzer-grid"
        id="analyzer"
    >


        <!-- INPUT -->

        <div class="analyzer-card">


            <div class="card-title">

                <div>

                    <h3>
                        Analyze Text
                    </h3>

                    <p>
                        Enter text and generate a sentiment prediction.
                    </p>

                </div>

                <i class="fa-solid fa-magnifying-glass-chart"></i>

            </div>


            <form
                method="POST"
                id="analysisForm"
            >


                <textarea
                    id="textInput"
                    name="text"
                    maxlength="5000"
                    placeholder="Enter text for sentiment analysis..."
                >{{ text }}</textarea>


                <div class="input-footer">


                    <div class="samples">


                        <button
                            type="button"
                            class="sample"
                            onclick="sampleText('I am extremely happy with this product. The quality is excellent.')"
                        >

                            Positive Example

                        </button>


                        <button
                            type="button"
                            class="sample"
                            onclick="sampleText('I am very disappointed with the service. It was a terrible experience.')"
                        >

                            Negative Example

                        </button>


                        <button
                            type="button"
                            class="sample"
                            onclick="sampleText('The product was delivered today.')"
                        >

                            Neutral Example

                        </button>


                    </div>


                    <span
                        id="counter"
                        style="
                        color:var(--muted);
                        font-size:9px;"
                    >

                        0 / 5000

                    </span>


                </div>


                <button
                    type="submit"
                    class="analyze-button"
                >

                    <i class="fa-solid fa-wand-magic-sparkles"></i>

                    Analyze Sentiment

                </button>


            </form>


            {% if error %}

            <div class="error-message">

                <i
                    class="fa-solid fa-circle-exclamation"
                ></i>

                {{ error }}

            </div>

            {% endif %}


        </div>


        <!-- RESULT -->

        <div class="analyzer-card">


            <div class="card-title">

                <div>

                    <h3>
                        Prediction Result
                    </h3>

                    <p>
                        Latest model classification
                    </p>

                </div>

                <i class="fa-solid fa-brain"></i>

            </div>


            <div class="result">


                {% if result %}


                <div
                    class="result-symbol {{ result.class }}"
                >

                    {{ result.emoji }}

                </div>


                <div
                    class="result-title {{ result.class }}"
                >

                    {{ result.label }}

                </div>


                <div class="result-description">

                    {{ result.description }}

                </div>


                {% if confidence is not none %}


                <div class="confidence-box">


                    <div class="confidence-header">

                        <span>
                            Confidence
                        </span>

                        <strong>
                            {{ confidence }}%
                        </strong>

                    </div>


                    <div class="progress">

                        <div
                            class="progress-value"
                        ></div>

                    </div>


                </div>


                {% endif %}


                {% else %}


                <div
                    class="result-symbol"
                    style="
                    background:var(--surface-2);
                    color:var(--muted);"
                >

                    <i class="fa-solid fa-brain"></i>

                </div>


                <div
                    class="result-title"
                    style="font-size:19px;"
                >

                    Ready for Analysis

                </div>


                <div class="result-description">

                    The prediction result will appear here.

                </div>


                {% endif %}


            </div>


        </div>


    </div>


    <!-- =================================================
         HISTORY
    ================================================== -->

    <div
        class="history-card"
        id="history"
    >


        <div class="history-header">


            <div>

                <h3>
                    Recent Analysis
                </h3>

                <p>
                    Latest sentiment prediction activity
                </p>

            </div>


            <button
                class="clear-button"
                onclick="clearHistory()"
            >

                <i class="fa-solid fa-trash"></i>

                Clear

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


                <div class="history-hide">

                    {% if item.confidence %}

                        {{ item.confidence }}%

                    {% else %}

                        —

                    {% endif %}

                </div>


                <div class="history-hide">

                    {{ item.time }}

                </div>


            </div>


            {% endfor %}


        {% else %}


        <div
            style="
            text-align:center;
            padding:30px;
            color:var(--muted);
            font-size:10px;"
        >

            <i
                class="fa-solid fa-clock-rotate-left"
                style="
                font-size:22px;
                margin-bottom:8px;"
            ></i>

            <br>

            No analysis history available.

        </div>


        {% endif %}


    </div>


</section>


<div class="footer">

    Sentiment Analysis
    &nbsp;•&nbsp;
    Flask
    &nbsp;•&nbsp;
    TF-IDF
    &nbsp;•&nbsp;
    Multinomial Naive Bayes

</div>


</main>

</div>


<!-- =====================================================
     JAVASCRIPT
===================================================== -->

<script>


/* =========================================================
   DASHBOARD DATA
========================================================= */

const data = {

    positive:
        {{ dashboard.positive }},

    negative:
        {{ dashboard.negative }},

    neutral:
        {{ dashboard.neutral }}

};


/* =========================================================
   THEME
========================================================= */

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
        "sentiment-theme",
        theme
    );

}


/* Load theme */

const savedTheme =
    localStorage.getItem(
        "sentiment-theme"
    );


if (savedTheme) {

    setTheme(savedTheme);

}


/* =========================================================
   SAMPLE TEXT
========================================================= */

function sampleText(text) {

    const input =
        document.getElementById(
            "textInput"
        );

    input.value = text;

    updateCounter();

    input.focus();

}


/* =========================================================
   CHARACTER COUNTER
========================================================= */

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


/* =========================================================
   DOUGHNUT CHART
========================================================= */

new Chart(

    document.getElementById(
        "donutChart"
    ),

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

                    data.positive,

                    data.negative,

                    data.neutral

                ],

                backgroundColor: [

                    "#10b981",

                    "#ef4444",

                    "#f59e0b"

                ],

                borderWidth: 0,

                hoverOffset: 7

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

                        usePointStyle: true,

                        padding: 15,

                        font: {
                            size: 10
                        }

                    }

                }

            }

        }

    }

);


/* =========================================================
   BAR CHART
========================================================= */

new Chart(

    document.getElementById(
        "barChart"
    ),

    {

        type: "bar",

        data: {

            labels: [

                "Positive",

                "Negative",

                "Neutral"

            ],

            datasets: [{

                data: [

                    data.positive,

                    data.negative,

                    data.neutral

                ],

                backgroundColor: [

                    "#10b981",

                    "#ef4444",

                    "#f59e0b"

                ],

                borderRadius: 7,

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

                        precision: 0,

                        font: {
                            size: 9
                        }

                    },

                    grid: {

                        color:
                            "rgba(148,163,184,.12)"

                    }

                },

                x: {

                    grid: {
                        display: false
                    },

                    ticks: {

                        font: {
                            size: 9
                        }

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


/* =========================================================
   FORM LOADING
========================================================= */

document
    .getElementById(
        "analysisForm"
    )
    .addEventListener(
        "submit",
        function() {

            const button =
                this.querySelector(
                    ".analyze-button"
                );

            button.innerHTML =
                '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

            button.disabled = true;

        }
    );


/* =========================================================
   NAVIGATION
========================================================= */

function goTo(id) {

    document
        .getElementById(id)
        .scrollIntoView({

            behavior:
                "smooth"

        });

}


/* =========================================================
   CLEAR HISTORY
========================================================= */

async function clearHistory() {

    const confirmClear =
        confirm(
            "Are you sure you want to clear the analysis history?"
        );


    if (!confirmClear) {

        return;

    }


    try {

        await fetch(
            "/api/clear",
            {
                method:
                    "POST"
            }
        );

        location.reload();

    }

    catch (error) {

        console.error(error);

    }

}


/* =========================================================
   CTRL + ENTER
========================================================= */

document
    .getElementById(
        "textInput"
    )
    .addEventListener(
        "keydown",
        function(event) {

            if (
                event.ctrlKey &&
                event.key === "Enter"
            ) {

                document
                    .getElementById(
                        "analysisForm"
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
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)

    print(
        "                 SENTIMENT ANALYSIS"
    )

    print(
        "              Professional Dashboard"
    )

    print("=" * 65)

    if MODEL_STATUS:

        print(
            "✓ Model loaded successfully"
        )

        print(
            "✓ Vectorizer loaded successfully"
        )

    else:

        print(
            "✗ Model loading failed"
        )

        print(
            f"Reason: {MODEL_ERROR}"
        )

    print("=" * 65)

    print(
        "Application URL:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 65 + "\n")


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
