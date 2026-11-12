# Sentiment Analysis Using NLP & Machine Learning

**An end-to-end text sentiment classification web application built with Python, Natural Language Processing (NLP), scikit-learn, and Flask.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Application-black?logo=flask)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)](https://scikit-learn.org/)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://sentiment-analysis-livid-nine.vercel.app/)

**[🚀 Live Demo](https://sentiment-analysis-livid-nine.vercel.app/)** · **[💻 Source Code](https://github.com/yashchaudhari372-lab/Sentiment_Analysis)**

---

## 📌 Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) project that analyzes written text and predicts its sentiment using a trained machine learning model. It demonstrates an end-to-end workflow: preparing text, converting it into numerical features, making predictions, and serving those predictions through a Flask web application.

The project can illustrate how organizations extract useful insights from customer reviews, feedback, and other text-based interactions..

## ✨ Key Features

- **Text-based sentiment prediction:** Enter text and obtain a model-generated sentiment classification.
- **NLP processing:** Uses a trained text vectorizer to transform input into machine-readable features.
- **Machine learning inference:** Loads a saved classification model for predictions.
- **Web interface:** Flask-powered application for interacting with the model.
- **Saved model artifacts:** Reuses the trained model and vectorizer without retraining on every request.
- **Online demo:** Explore the deployed application using the live demo link above.

## 🛠️ Technology Stack

| Category | Technologies |
| --- | --- |
| Programming language | Python |
| Machine learning | scikit-learn, saved classification model |
| Natural language processing | Text preprocessing and vectorization |
| Web framework | Flask |
| Model persistence | Pickle (`.pkl`) artifacts |
| Deployment | Live web application on Vercel |
| Version control | Git and GitHub |

## ⚙️ How It Works

```text
User enters text
       ↓
Text is prepared for prediction
       ↓
Saved vectorizer transforms the text
       ↓
Trained machine learning model predicts sentiment
       ↓
Flask application displays the result
```

### Machine Learning Workflow

1. **Text preparation:** Prepare the input text for model inference.
2. **Feature extraction:** Convert text into numerical features using the saved vectorizer.
3. **Model prediction:** Pass the resulting features into the saved classification model.
4. **Result presentation:** Display the predicted sentiment through the web interface.

> **Note:** The repository includes inference artifacts, but does not currently expose a separate training notebook or dataset. The exact training preprocessing steps, algorithm, and dataset should be documented once the training code is available.

## 📊 Model Performance

The project's previously reported evaluation results are:

| Metric | Reported Result |
| --- | ---: |
| Accuracy | **85%** |
| F1-score | **0.85** |
| Test samples | **10,000** |

These are previously reported project metrics; the current repository does not include the evaluation script or classification report needed to independently reproduce them. For reproducibility, consider adding your test-set methodology, confusion matrix, and classification report.

## 📁 Repository Structure

```text
Sentiment_Analysis/
├── app.py             # Flask application and prediction interface
├── model (1).pkl      # Saved trained machine learning model
├── vectorizer.pkl     # Saved text vectorizer
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## 🚀 Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/yashchaudhari372-lab/Sentiment_Analysis.git
cd Sentiment_Analysis
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```

Open the local URL printed in your terminal (commonly `http://127.0.0.1:5000/`). If your app uses a different port, use that port instead.

> **Security note:** Only load `.pkl` files from trusted sources. Pickle deserialization can execute arbitrary code.

## 💡 Example Use Cases

- **Customer feedback:** Identify sentiment in customer comments.
- **Product reviews:** Analyze reactions to products or services.
- **Social media:** Explore patterns in text-based opinions.
- **Brand monitoring:** Help categorize feedback for further analysis.
- **Business analytics:** Support sentiment-based reporting and decision-making.

## 🔮 Potential Improvements

- Add training and evaluation notebooks for reproducibility.
- Include a confusion matrix and class-level precision, recall, and F1-scores.
- Add tests for empty input, unusual characters, and long text.
- Improve model interpretability and confidence calibration.
- Introduce batch analysis for multiple reviews.
- Compare classical machine learning models with transformer-based approaches.

## 👨‍💻 Developer

**Yash Chaudhari**  
Aspiring Data Scientist | Data Analyst | Machine Learning & NLP Enthusiast.

- **GitHub:** [yashchaudhari372-lab](https://github.com/yashchaudhari372-lab)
- **LinkedIn:** [yashchaudhari08](https://www.linkedin.com/in/yashchaudhari08/)

## ⭐ Support

If you find this project useful, consider starring the repository. Feedback, suggestions, and contributions are welcome!
