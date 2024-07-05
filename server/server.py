from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from PIL import Image
import glob

app = Flask(__name__)
CORS(app)

# Global variables for machine learning model
model = None
trained = False

MODEL_PATH = './model/random_forest_model.pkl'
UPLOAD_FOLDER = './data'

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    
    return jsonify({'message': 'File uploaded successfully'}), 200

# Train endpoint
@app.route('/train', methods=['POST'])
def train_model():
    global model, trained

    if trained:
        return jsonify({'message': 'Model already trained'}), 200

    X_train, y_train = load_images_and_labels(UPLOAD_FOLDER)
    
    if X_train is None or y_train is None:
        return jsonify({'error': 'No data to train on'}), 400

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Ensure the model directory exists
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))

    joblib.dump(model, MODEL_PATH)
    trained = True
    
    return jsonify({'message': 'Model trained successfully'}), 200

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict_image():
    global model, trained

    if not trained:
        return jsonify({'error': 'Model not trained yet'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = preprocess_image(file)
    
    prediction = model.predict([image])[0]
    
    return jsonify({'prediction': 'cancerous' if prediction == 1 else 'noncancerous'}), 200

# Load model if exists
def load_model():
    global model, trained
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        trained = True

# Helper function to load images and labels
def load_images_and_labels(folder_path):
    images = []
    labels = []
    
    benign_files = glob.glob(os.path.join(folder_path, 'benign', '*.jpg'))
    malignant_files = glob.glob(os.path.join(folder_path, 'malignant', '*.jpg'))
    
    for file in benign_files:
        img = preprocess_image(file)
        images.append(img)
        labels.append(0)
    
    for file in malignant_files:
        img = preprocess_image(file)
        images.append(img)
        labels.append(1)
    
    if images and labels:
        images = np.array(images)
        labels = np.array(labels)
        return images, labels
    else:
        return None, None

# Helper function to preprocess images
def preprocess_image(file):
    img = Image.open(file)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    return img.flatten()

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
