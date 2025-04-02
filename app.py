from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import dlib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datetime import datetime
import google.generativeai as genai
import os

app = Flask(__name__)

# Initialize Gemini API
#Gemini
class AIInsightGenerator:
    def __init__(self, api_key):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            print(f"AI Insight Generator initialization error: {e}")
            self.model = None

    def generate_insights(self, face_analysis):
        if not self.model:
            return "AI insights unavailable"

        try:
            prompt = f"""
            You are an advanced facial health analyst. Interpret the following metrics:

            Symmetry Score: {face_analysis.get('symmetry_score', 'N/A')}
            Proportion Score: {face_analysis.get('proportion_score', 'N/A')}
            Alignment Score: {face_analysis.get('alignment_score', 'N/A')}
            Paralysis Detection: {face_analysis.get('paralysis_detection', 'N/A')}

            Provide health insights with sections wrapped in <h3>, <p>, and <ul> tags... and add style text-align to justify.
            """
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"AI insight generation error: {e}")
            return "Error generating AI insights"

# Face Detector using dlib
class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        detected_faces = []
        for face in faces:
            x, y = face.left(), face.top()
            w, h = face.width(), face.height()
            face_roi = frame[y:y + h, x:x + w]
            detected_faces.append((x, y, x + w, y + h, face_roi))
        return detected_faces

# Facial Analysis Model
class FacialAnalysisModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

# Utility Functions
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_symmetry(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    left_half = gray[:, :width // 2]
    right_half = gray[:, width // 2:]
    left_mean = np.mean(left_half)
    right_mean = np.mean(right_half)
    return abs(left_mean - right_mean)

def calculate_proportion(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    return width / height if height != 0 else 0

def calculate_alignment(face_bbox, image_width, image_height):
    x1, y1, x2, y2 = face_bbox
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2
    image_center_x = image_width / 2
    image_center_y = image_height / 2

    horizontal_diff = abs(face_center_x - image_center_x) / (image_width / 2)
    vertical_diff = abs(face_center_y - image_center_y) / (image_height / 2)

    alignment_score = 1 - (horizontal_diff + vertical_diff) / 2
    return alignment_score * 100

def detect_paralysis(face_image):
    symmetry_threshold = 10
    symmetry_score = calculate_symmetry(face_image)
    return "Possible Paralysis Detected" if symmetry_score > symmetry_threshold else "No Paralysis Detected"



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load the image
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Detect faces
        face_detector = FaceDetector()
        faces = face_detector.detect_faces(frame)
        if not faces:
            return jsonify({'message': 'No faces detected'}), 200

        # Analyze first detected face
        x1, y1, x2, y2, face_roi = faces[0]
        symmetry_score = calculate_symmetry(face_roi)
        proportion_score = calculate_proportion(face_roi)
        paralysis_detection = detect_paralysis(face_roi)
        image_height, image_width = frame.shape[:2]
        alignment_score = calculate_alignment((x1, y1, x2, y2), image_width, image_height)

        # Generate AI insights
        ai_generator = AIInsightGenerator("AIzaSyDx0R9BZPuPjXqR5drRGzC2IZm_5w2CURk")
        face_analysis = {
            'symmetry_score': symmetry_score,
            'proportion_score': proportion_score,
            'paralysis_detection': paralysis_detection,
            'alignment_score': alignment_score
        }
        ai_insights = ai_generator.generate_insights(face_analysis)

        return jsonify({
            'symmetry_score': symmetry_score,
            'proportion_score': proportion_score,
            'paralysis_detection': paralysis_detection,
            'alignment_score': alignment_score,
            'ai_insights': ai_insights
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
