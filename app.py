import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    words = [word for word in tokens if word not in stop_words]
    return " ".join([lemmatizer.lemmatize(word) for word in words])


sample_text = "This movie was absolutely wonderful and touching!"
cleaned = preprocess(sample_text)
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)

print(prediction)