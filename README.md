# Intelligent Q&A Classifier (NLP)

A Machine Learning application that classifies user questions into domain categories (Technology, Science, Life Arts) with **84% accuracy**. Built using the Google Quest dataset to simulate automated IT/Support ticket routing.

## Features
- **Real-time Classification:** Classifies text inputs instantly using a pre-trained SVM model.
- **Intent Detection:** "Pro" logic layer identifies if a query is a "How-to", "Troubleshooting", or "General" question.
- **Performance:** Achieved 84% Weighted F1-Score on the validation set.

## Tech Stack
- **Python**
- **Scikit-Learn:** TF-IDF Vectorization & Linear SVC
- **Streamlit:** Interactive Web Interface
- **Pandas/NLTK:** Data Processing & NLP

## Dataset
Used the **Google Quest Challenge** dataset, training on 6,000+ real-world labeled questions.

## How to Run Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/intelligent-ticket-classifier.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the app:
   ```bash
   streamlit run app.py
