from flask import Flask, request, render_template
import joblib
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)
model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    data = vectorizer.transform([news])
    prediction = model.predict(data)[0]
    result = "Fake News" if prediction == 1 else "Real News"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
