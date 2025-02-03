# face_analysis.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import dlib
import google.generativeai as genai
import time
from tenacity import retry, wait_exponential, stop_after_attempt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return [(face.left(), face.top(), face.right(), face.bottom()) for face in faces]
    
    def get_landmarks(self, frame, face_rect):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.predictor(gray, face_rect)

class AIInsightGenerator:
    def __init__(self, api_key):
        try:
            genai.configure(api_key="AIzaSyCdDTrxnZhbyRkG7KREC1SoeG5gGC1O62A")
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            print(f"AI Insight Generator error: {e}")
            self.model = None
    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), 
           stop=stop_after_attempt(5))
    def generate_insights(self, analysis):

        try:
            print(f"API Request - Time: {time.time()}, Analysis Hash: {hash(str(analysis))}")
            response = self.model.generate_content(prompt)
            print(f"API Response Headers: {response._result.response.headers}")
            prompt = f"""
            Analyze these facial metrics:
            - Symmetry: {analysis.get('symmetry_score', 'N/A')}
            - Proportion: {analysis.get('proportion_score', 'N/A')}
            - Alignment: {analysis.get('alignment_score', 'N/A')}
            - Paralysis: {analysis.get('paralysis_detection', 'N/A')}
            
            Provide concise health recommendations in bullet points.
            """
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Full Error Details: {dir(e)}")
            print(f"HTTP Status Code: {e.status_code}")
            print(f"Error Details: {e._cause}")
            return "AI insights temporarily unavailable"

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
        return self.classifier(self.features(x))

class MultiFaceAnalyzer:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.ai_generator = AIInsightGenerator("AIzaSyCdDTrxnZhbyRkG7KREC1SoeG5gGC1O62A")
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        model = FacialAnalysisModel()
        # Load your trained weights here
        # model.load_state_dict(torch.load("model_weights.pth"))
        model.eval()
        return model

    def process_frame(self, frame):
        faces = self.face_detector.detect_faces(frame)
        analyses = []
        
        for (x1, y1, x2, y2) in faces:
            face_roi = frame[y1:y2, x1:x2]
            analysis = self.analyze_face(face_roi)
            analyses.append({
                "coordinates": (x1, y1, x2, y2),
                "metrics": {
                    "symmetry_score": analysis["symmetry_score"],
                    "proportion_score": analysis["proportion_score"],
                    "alignment_score": analysis["alignment_score"],
                    "paralysis_detection": analysis["paralysis_detection"]
                },
                "insights": analysis["ai_insights"]
            })
            frame = self.draw_analysis(frame, x1, y1, x2, y2, analysis)
        
        return {"faces": analyses, "frame": frame}

    def analyze_face(self, face_roi):
        with torch.no_grad():
            face_tensor = self.transform(face_roi).unsqueeze(0)
            outputs = self.model(face_tensor)
        
        analysis = {
            "symmetry_score": self.calculate_symmetry(face_roi),
            "proportion_score": self.calculate_proportion(face_roi),
            "alignment_score": self.calculate_alignment(face_roi),
            "paralysis_detection": self.detect_paralysis(face_roi),
            "raw_outputs": outputs.numpy().tolist()
        }
        analysis["ai_insights"] = self.ai_generator.generate_insights(analysis)
        return analysis

    # Keep the original analysis methods from your code
    def calculate_symmetry(self, face_image):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        mid = gray.shape[1] // 2
        return abs(np.mean(gray[:, :mid]) - np.mean(gray[:, mid:]))

    def calculate_proportion(self, face_image):
        h, w = face_image.shape[:2]
        return w / h if h != 0 else 0

    def calculate_alignment(self, face_image):
        return np.random.uniform(0, 1)  # Replace with actual logic

    def detect_paralysis(self, face_image):
        return "No paralysis detected" if self.calculate_symmetry(face_image) < 10 else "Possible paralysis"

    def draw_analysis(self, frame, x1, y1, x2, y2, analysis):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Sym: {analysis['symmetry_score']:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return frame