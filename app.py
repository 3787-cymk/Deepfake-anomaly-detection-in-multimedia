import os
import uuid
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import random


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png', 'webp'}

app = Flask(__name__, static_folder="static")
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


MODEL_PATH = os.path.join("Saved Models", "models", "CNN_RNN", "face_detction_CNN_RNN.h5")

def create_model():
    input_shape = (20, 2048)
    mask_input = layers.Input(shape=(20, 1))
    inputs = layers.Input(shape=input_shape)

   
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    
    x = layers.LSTM(64)(x)
    
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model([inputs, mask_input], outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
else:
    model = create_model()
    print("⚠️ Model not found, a new model is created!")

feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

def extract_features(video_path, target_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    start_frame = max(0, total_frames // 4)
    end_frame = min(total_frames, (total_frames * 3) // 4)
    frames_to_sample = np.linspace(start_frame, end_frame, target_frames, dtype=int)
    
    frames = []
    for frame_idx in frames_to_sample:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)
        frames.append(frame)
            
    cap.release()

    
    while len(frames) < target_frames:
        frames.append(np.zeros((224, 224, 3)))
    
    frames = np.array(frames)
    extracted_features = feature_extractor.predict(frames, batch_size=4)
    extracted_features = extracted_features.reshape((1, target_frames, 2048))
    
   
    mask = np.ones((1, target_frames, 1))
    if len(frames) < target_frames:
        mask[0, len(frames):, 0] = 0
    
    return extracted_features, mask

def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'webp'}

def analyze_image(image_path):
    # Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Unable to read image"
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    # Extract features using InceptionV3
    features = feature_extractor.predict(img, verbose=0)
    
    # Reshape features to match model input shape
    features = np.tile(features, (1, 20, 1))
    mask = np.ones((1, 20, 1))
    
    # Make prediction
    prediction = model.predict([features, mask], verbose=0)[0][0]
    print(f"🔍 Raw Image Prediction: {prediction:.3f}")
    
    return format_prediction(prediction)

def format_prediction(prediction):
    if prediction >= 0.85:  
        return f"FAKE (Confidence: {prediction:.2f})"
    elif prediction <= 0.15:  
        return f"REAL (Confidence: {(1-prediction):.2f})"
    else:
        confidence = abs(0.5 - prediction) * 2
        if prediction > 0.5:
            return f"LIKELY FAKE (Confidence: {confidence:.2f})"
        else:
            return f"LIKELY REAL (Confidence: {confidence:.2f})"

def predict_fake(file_path):
    if model is None:
        return "MODEL MISSING"

    if is_image_file(file_path):
        return analyze_image(file_path)
    
    # Video analysis
    features, mask = extract_features(file_path)
    prediction = model.predict([features, mask], verbose=0)[0][0]
    print(f"🔍 Raw Video Prediction: {prediction:.3f}")
    
    return format_prediction(prediction)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part", "error")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{ext}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            flash("File successfully uploaded", "success")
            return redirect(url_for("upload_file", filename=unique_filename))
        else:
            flash("File type not allowed", "error")
            return redirect(request.url)
    filename = request.args.get("filename")
    return render_template("upload.html", filename=filename)

@app.route("/uploads/<filename>")
def display_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/analyze/<filename>")
def sequence_prediction(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path):
        flash("File not found for analysis", "error")
        return redirect(url_for("upload_file"))
    prediction = predict_fake(video_path)
    flash(f"Analysis Result: {prediction}", "info")
    return render_template("upload.html", filename=filename, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5001)