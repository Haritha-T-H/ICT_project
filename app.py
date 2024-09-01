from flask import Flask, render_template, request
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# Load your model and vectorizer
model = pickle.load(open('best_xgb.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

# Initialize NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d', ' ', text)  # Remove digits
    words = word_tokenize(text)  # Tokenize the text
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]  # Stemming, lemmatization
    return ' '.join(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('prediction.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        message = request.form['message']
        processed_message = preprocess_text(message)
        message_tfidf = vectorizer.transform([processed_message])
        prediction = model.predict(message_tfidf)

        result = 'Spam' if prediction[0] == 1 else 'Ham'

        return render_template('result.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
