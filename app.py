# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
svm_model = joblib.load('svm_classifier.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Transform the data using the tfidf_vectorizer
    transformed_data = tfidf_vectorizer.transform([data['Article']])
    # Use the SVM model to make a prediction
    prediction = svm_model.predict(transformed_data)
    return jsonify(prediction[0])

if __name__ == '__main__':
    app.run(port=5000)
