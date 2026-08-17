import os
import pickle
import warnings
from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model (1).pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentiment Intelligence Dashboard</title>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
:root{
    --bg:#080b16;
    --card:rgba(18,24,43,.78);
    --card2:rgba(24,31,54,.92);
    --text:#f7f8ff;
    --muted:#aab3ca;
    --border:rgba(255,255,255,.10);
    --primary:#7c5cff;
    --secondary:#00d4ff;
    --accent:#ff4ecd;
    --success:#20e3a2;
    --danger:#ff5d7a;
    --shadow:0 24px 70px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{
    margin:0; color:var(--text);
    font-family:Inter,system-ui,sans-serif;
    background:
      radial-gradient(circle at 10% 10%,rgba(124,92,255,.24),transparent 30%),
      radial-gradient(circle at 90% 20%,rgba(0,212,255,.18),transparent 28%),
      radial-gradient(circle at 50% 100%,rgba(255,78,205,.13),transparent 30%),
      var(--bg);
    min-height:100vh;
    overflow-x:hidden;
}
body:before{
    content:"";position:fixed;inset:0;pointer-events:none;opacity:.22;
    background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),
                     linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
    background-size:42px 42px;
}
.container{width:min(1180px,92%);margin:auto}
.nav{
    display:flex;justify-content:space-between;align-items:center;
    padding:26px 0; position:relative;z-index:2
}
.brand{display:flex;gap:12px;align-items:center;font-family:"Space Grotesk";font-weight:700;font-size:20px}
.logo{
    width:42px;height:42px;border-radius:14px;
    display:grid;place-items:center;
    background:linear-gradient(135deg,var(--primary),var(--accent),var(--secondary));
    box-shadow:0 0 32px rgba(124,92,255,.45);
}
.theme-wrap{display:flex;gap:8px;align-items:center}
.theme-btn{
    border:1px solid var(--border);background:rgba(255,255,255,.06);
    color:var(--text);border-radius:12px;padding:9px 12px;cursor:pointer;
    transition:.25s
}
.theme-btn:hover{transform:translateY(-2px);border-color:rgba(255,255,255,.25)}
.hero{padding:38px 0 25px;position:relative}
.badge{
    display:inline-flex;padding:8px 12px;border-radius:999px;
    background:rgba(32,227,162,.10);border:1px solid rgba(32,227,162,.25);
    color:var(--success);font-size:12px;font-weight:700;letter-spacing:.5px
}
h1{
    font-family:"Space Grotesk";font-size:clamp(38px,6vw,70px);
    line-height:1.02;margin:18px 0 14px;max-width:850px;
    background:linear-gradient(90deg,#fff,var(--secondary),#fff,var(--accent));
    background-size:250% auto;-webkit-background-clip:text;background-clip:text;color:transparent;
    animation:shine 7s linear infinite
}
@keyframes shine{to{background-position:250% center}}
.hero p{color:var(--muted);max-width:720px;font-size:16px;line-height:1.8}
.grid{display:grid;grid-template-columns:1.08fr .92fr;gap:22px;margin:22px 0}
.card{
    background:linear-gradient(145deg,rgba(255,255,255,.07),rgba(255,255,255,.025));
    border:1px solid var(--border);border-radius:24px;padding:25px;
    box-shadow:var(--shadow);backdrop-filter:blur(18px);
    animation:rise .7s ease both
}
@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
.card h2{font-family:"Space Grotesk";margin:0 0 8px;font-size:21px}
.sub{color:var(--muted);font-size:13px;margin-bottom:20px}
textarea{
    width:100%;min-height:205px;resize:vertical;padding:18px;
    border-radius:18px;border:1px solid var(--border);
    background:rgba(4,7,16,.62);color:var(--text);
    outline:none;font:500 15px/1.7 Inter;transition:.25s
}
textarea:focus{border-color:var(--secondary);box-shadow:0 0 0 4px rgba(0,212,255,.08)}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin-top:15px}
button.primary{
    border:0;padding:13px 19px;border-radius:14px;color:white;
    background:linear-gradient(135deg,var(--primary),var(--accent));
    font-weight:800;cursor:pointer;box-shadow:0 10px 28px rgba(124,92,255,.25);
    transition:.25s
}
button.primary:hover{transform:translateY(-2px);box-shadow:0 14px 34px rgba(124,92,255,.35)}
button.secondary{
    border:1px solid var(--border);padding:13px 19px;border-radius:14px;
    background:rgba(255,255,255,.06);color:var(--text);font-weight:700;cursor:pointer
}
.result{
    min-height:205px;display:grid;place-items:center;text-align:center;
    border-radius:18px;background:rgba(4,7,16,.42);border:1px solid var(--border)
}
.result .emoji{font-size:48px;filter:drop-shadow(0 0 18px rgba(255,255,255,.18))}
.label{font-family:"Space Grotesk";font-size:30px;font-weight:800;margin-top:6px}
.conf{color:var(--muted);margin-top:7px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:20px}
.stat{padding:17px;border:1px solid var(--border);border-radius:18px;background:rgba(255,255,255,.04)}
.stat b{font-size:24px;font-family:"Space Grotesk"}
.stat span{display:block;color:var(--muted);font-size:12px;margin-top:5px}
.chart-card{margin:22px 0}
.chart-box{height:330px;position:relative}
footer{text-align:center;color:var(--muted);font-size:12px;padding:30px 0 45px}
.loader{
    width:23px;height:23px;border:3px solid rgba(255,255,255,.18);
    border-top-color:var(--secondary);border-radius:50%;animation:spin .8s linear infinite
}
@keyframes spin{to{transform:rotate(360deg)}}
.themes{display:none;position:absolute;right:0;top:75px;background:var(--card2);
border:1px solid var(--border);border-radius:18px;padding:10px;box-shadow:var(--shadow);z-index:10}
.themes.open{display:flex;gap:8px}
.dot{width:24px;height:24px;border-radius:50%;border:2px solid rgba(255,255,255,.5);cursor:pointer}
@media(max-width:850px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr}}
</style>
</head>

<body>
<div class="container">
<nav class="nav">
    <div class="brand"><div class="logo">AI</div> Sentiment Intelligence</div>
    <div style="position:relative">
        <div class="theme-wrap">
            <button class="theme-btn" onclick="toggleThemes()">🎨 Theme</button>
        </div>
        <div class="themes" id="themes">
            <div class="dot" title="Aurora" style="background:linear-gradient(135deg,#7c5cff,#00d4ff)" onclick="theme('aurora')"></div>
            <div class="dot" title="Sunset" style="background:linear-gradient(135deg,#ff7a18,#ff3d81)" onclick="theme('sunset')"></div>
            <div class="dot" title="Ocean" style="background:linear-gradient(135deg,#00c6ff,#0072ff)" onclick="theme('ocean')"></div>
            <div class="dot" title="Emerald" style="background:linear-gradient(135deg,#00b09b,#96c93d)" onclick="theme('emerald')"></div>
            <div class="dot" title="Royal" style="background:linear-gradient(135deg,#8e2de2,#4a00e0)" onclick="theme('royal')"></div>
        </div>
    </div>
</nav>

<section class="hero">
    <span class="badge">● MODEL ONLINE · MULTINOMIAL NAIVE BAYES</span>
    <h1>Turn text into<br>clear sentiment insights.</h1>
    <p>Analyze customer reviews, feedback and comments with your trained NLP model. Get an instant sentiment prediction, confidence score and professional visual analysis.</p>
</section>

<section class="grid">
    <div class="card">
        <h2>📝 Text Analyzer</h2>
        <div class="sub">Enter a review or any English text.</div>
        <textarea id="text" placeholder="Example: The product quality is excellent and I really enjoyed using it..."></textarea>
        <div class="controls">
            <button class="primary" onclick="analyze()">Analyze Sentiment</button>
            <button class="secondary" onclick="clearAll()">Clear</button>
        </div>
    </div>

    <div class="card">
        <h2>📊 Prediction</h2>
        <div class="sub">Live result from your model + TF-IDF vectorizer.</div>
        <div class="result" id="result">
            <div>
                <div class="emoji">🔮</div>
                <div class="label">Waiting for text</div>
                <div class="conf">Your prediction will appear here</div>
            </div>
        </div>
        <div class="stats">
            <div class="stat"><b id="sentimentStat">—</b><span>SENTIMENT</span></div>
            <div class="stat"><b id="confidenceStat">—</b><span>CONFIDENCE</span></div>
            <div class="stat"><b id="wordsStat">0</b><span>WORDS</span></div>
        </div>
    </div>
</section>

<div class="card chart-card">
    <h2>📈 Probability Analysis</h2>
    <div class="sub">Model probability distribution for the latest prediction.</div>
    <div class="chart-box"><canvas id="probChart"></canvas></div>
</div>

<footer>Built with Flask · TF-IDF · Multinomial Naive Bayes · Chart.js</footer>
</div>

<script>
let chart = null;

const themes = {
  aurora:{primary:"#7c5cff",secondary:"#00d4ff",accent:"#ff4ecd"},
  sunset:{primary:"#ff7a18",secondary:"#ff3d81",accent:"#ffd166"},
  ocean:{primary:"#00c6ff",secondary:"#0072ff",accent:"#00f5d4"},
  emerald:{primary:"#00b09b",secondary:"#96c93d",accent:"#00e5a8"},
  royal:{primary:"#8e2de2",secondary:"#4a00e0",accent:"#c77dff"}
};

function theme(name){
    const t=themes[name];
    document.documentElement.style.setProperty("--primary",t.primary);
    document.documentElement.style.setProperty("--secondary",t.secondary);
    document.documentElement.style.setProperty("--accent",t.accent);
    localStorage.setItem("sentiment-theme",name);
    document.getElementById("themes").classList.remove("open");
}
function toggleThemes(){document.getElementById("themes").classList.toggle("open")}
const saved=localStorage.getItem("sentiment-theme"); if(saved) theme(saved);

function clearAll(){
    document.getElementById("text").value="";
    document.getElementById("wordsStat").textContent="0";
    document.getElementById("sentimentStat").textContent="—";
    document.getElementById("confidenceStat").textContent="—";
    document.getElementById("result").innerHTML='<div><div class="emoji">🔮</div><div class="label">Waiting for text</div><div class="conf">Your prediction will appear here</div></div>';
    if(chart){chart.destroy();chart=null;}
}

async function analyze(){
    const text=document.getElementById("text").value.trim();
    if(!text){alert("Please enter some text first.");return;}

    const result=document.getElementById("result");
    result.innerHTML='<div class="loader"></div>';

    try{
        const response=await fetch("/predict",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({text})
        });
        const data=await response.json();
        if(!response.ok) throw new Error(data.error || "Prediction failed");

        const positive=data.sentiment.toLowerCase()==="positive";
        result.innerHTML=`<div>
            <div class="emoji">${positive?"😊":"😞"}</div>
            <div class="label">${data.sentiment}</div>
            <div class="conf">${data.confidence.toFixed(2)}% confidence</div>
        </div>`;

        document.getElementById("sentimentStat").textContent=data.sentiment;
        document.getElementById("confidenceStat").textContent=data.confidence.toFixed(1)+"%";
        document.getElementById("wordsStat").textContent=text.split(/\s+/).filter(Boolean).length;

        const labels=Object.keys(data.probabilities);
        const values=Object.values(data.probabilities);

        if(chart) chart.destroy();
        chart=new Chart(document.getElementById("probChart"),{
            type:"bar",
            data:{
                labels:labels.map(x=>x.toUpperCase()),
                datasets:[{
                    label:"Probability %",
                    data:values,
                    borderRadius:12,
                    borderWidth:1
                }]
            },
            options:{
                responsive:true,maintainAspectRatio:false,
                plugins:{legend:{display:false}},
                scales:{
                    x:{grid:{display:false},ticks:{color:"#aab3ca"}},
                    y:{beginAtZero:true,max:100,grid:{color:"rgba(255,255,255,.07)"},ticks:{color:"#aab3ca",callback:v=>v+"%"}}
                }
            }
        });
    }catch(err){
        result.innerHTML='<div><div class="emoji">⚠️</div><div class="label">Prediction Error</div><div class="conf">'+err.message+'</div></div>';
    }
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()

        if not text:
            return jsonify({"error": "Please provide text."}), 400

        # The uploaded vectorizer is a fitted TfidfVectorizer with 5,000 features.
        X = vectorizer.transform([text])

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        classes = [str(c) for c in model.classes_]
        probability_map = {
            label: round(float(prob) * 100, 2)
            for label, prob in zip(classes, probabilities)
        }

        confidence = float(max(probabilities)) * 100

        return jsonify({
            "sentiment": str(prediction),
            "confidence": round(confidence, 2),
            "probabilities": probability_map
        })

    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
