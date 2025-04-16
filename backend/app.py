from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to access the backend

# Load vectorizer and label encoder
try:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("✅ TF-IDF vectorizer and label encoder loaded successfully.")
except Exception as e:
    print(f"❌ Error loading vectorizer or label encoder: {e}")

# Load best model
model = None
try:
    with open("best_model.txt", "r") as f:
        best_model_name = f.read().strip()
        model_path = f"{best_model_name}_model.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅ Loaded best model: {best_model_name}")
    else:
        print(f"❌ Model file '{model_path}' not found.")
except Exception as e:
    print(f"❌ Error loading the best model: {e}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Fake News Detection API is running!"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if request.content_type != "application/json":
        return jsonify({"error": "Invalid content type. Expected 'application/json'"}), 415

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    try:
        input_text = [data["text"]]
        transformed_text = vectorizer.transform(input_text)
        prediction = model.predict(transformed_text)
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"prediction": prediction_label}), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
