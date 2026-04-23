# ================================
# PHISHING EMAIL DETECTOR (IMPROVED)
# ================================

import os
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# CLEAR TERMINAL (Windows)
# -------------------------------
os.system('cls')

print("="*50)
print(" PHISHING EMAIL DETECTION SYSTEM ")
print("="*50)

# -------------------------------
# LOAD DATASET
# -------------------------------
data = pd.read_csv("emails.csv")

# -------------------------------
# PREPROCESS FUNCTION
# -------------------------------
def preprocess(text):
    text = str(text).lower()
    
    # Replace URLs
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    
    # Remove special characters
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

data['text'] = data['text'].apply(preprocess)

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2)  # capture phrases like "click link"
)

X = vectorizer.fit_transform(data['text'])
y = data['label']

# -------------------------------
# TRAIN-TEST SPLIT (BALANCED)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# MODEL (BETTER THAN NAIVE BAYES)
# -------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -------------------------------
# PREDICTION & EVALUATION
# -------------------------------
y_pred = model.predict(X_test)

print("\n📊 MODEL PERFORMANCE")
print("-"*50)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# -------------------------------
# EMAIL PREDICTION FUNCTION
# -------------------------------
def predict_email(email_text):
    processed = preprocess(email_text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    return prediction

# -------------------------------
# USER INPUT TEST
# -------------------------------
print("\n" + "="*50)
print(" TEST YOUR EMAIL ")
print("="*50)

user_input = input("Enter email text: ")

result = predict_email(user_input)

print("\n🔍 Prediction:", result.upper())