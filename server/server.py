from flask import Flask, jsonify, request
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import glob

app = Flask(__name__)

# Global variables for machine learning model
model = None
X_train = None
y_train = None

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_image():
    global X_train, y_train
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file to a designated folder
    upload_folder = './data'  # Assuming your data folder is named 'data'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file.save(os.path.join(upload_folder, file.filename))
    
    # Process the uploaded file for training
    X_train, y_train = load_images_and_labels(upload_folder)
    
    return jsonify({'message': 'File uploaded successfully'}), 200

# Train endpoint
@app.route('/train', methods=['POST'])
def train_model():
    global model, X_train, y_train
    
    if X_train is None or y_train is None:
        return jsonify({'error': 'No data uploaded yet'}), 400
    
    # Train a simple classifier (Random Forest as an example)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    return jsonify({'message': 'Model trained successfully'}), 200

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict_image():
    global model
    
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Load and preprocess the uploaded image
    image = preprocess_image(file)
    
    # Make prediction
    prediction = model.predict([image])[0]
    
    return jsonify({'prediction': 'cancerous' if prediction == 1 else 'noncancerous'}), 200

# Helper function to load images and labels
def load_images_and_labels(folder_path):
    images = []
    labels = []
    
    cancerous_files = glob.glob(os.path.join(folder_path, 'cancerous', '*.jpg'))
    noncancerous_files = glob.glob(os.path.join(folder_path, 'noncancerous', '*.jpg'))
    
    for file in cancerous_files:
        img = preprocess_image(file)
        images.append(img)
        labels.append(1)  # Cancerous
    
    for file in noncancerous_files:
        img = preprocess_image(file)
        images.append(img)
        labels.append(0)  # Noncancerous
    
    return np.array(images), np.array(labels)

# Helper function to preprocess images
def preprocess_image(file):
    img = Image.open(file)
    img = img.resize((224, 224))  # Resize image if needed
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values
    return img.flatten()  # Flatten image to vector

if __name__ == '__main__':
    app.run(debug=True)

