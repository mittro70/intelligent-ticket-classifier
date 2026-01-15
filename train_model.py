import pandas as pd
import joblib
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv("train.csv")
df = df[['question_title', 'category']]
df.rename(columns={'question_title': 'text'}, inplace=True)

# Preprocessing
nltk.download('stopwords')
stop_words = stopwords.words('english')
df['clean_text'] = df['text'].apply(lambda x: str(x).lower())

# Train
X = df['clean_text']
y = df['category']
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words, max_features=5000)),
    ('clf', LinearSVC())
])
pipeline.fit(X, y)

# Save
joblib.dump(pipeline, 'quest_model.pkl')
print("Model trained and saved.")
