# 🎭 Deepfake Anomaly Detection in Multimedia

An AI-powered web application that detects whether uploaded images or videos are **REAL or FAKE (Deepfake)** using deep learning techniques.

---

## 🚀 Features

* 🔍 Detects deepfake content in **images and videos**
* 🎥 Supports video upload and frame-based analysis
* 🧠 Uses deep learning models for classification
* 🌐 Simple web interface using Flask
* 📊 Real-time prediction display

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS
* **Backend:** Flask (Python)
* **ML/DL:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Dataset:** DeepFake Detection Challenge (sample)

---

## 📂 Project Structure

```
Deepfake-anomaly-detection-in-multimedia/
│── app.py
│── requirements.txt
│── README.md
├── static/uploads/
├── templates/
└── model/
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/3787-cymk/Deepfake-anomaly-detection-in-multimedia.git
cd Deepfake-anomaly-detection-in-multimedia
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000
```

---

## 📸 How It Works

1. User uploads an image/video
2. Frames are extracted (for videos)
3. Preprocessing is applied
4. Model predicts REAL or FAKE
5. Result displayed on UI

---

## ⚠️ Important Notes

* Large files (videos) are not stored in the repository
* Upload your own test media in `static/uploads/`
* Model file may be provided separately due to size limits

---

## 📈 Future Improvements

* 🔊 Audio deepfake detection
* 📱 Mobile-friendly UI
* ⚡ Faster inference using optimized models
* ☁️ Deployment on cloud

---

## 👩‍💻 Author

**Archi Jain**
B.Tech AI | Deep Learning Enthusiast

---

## ⭐ If you like this project, give it a star!
