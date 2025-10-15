
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Flask 앱 생성
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads"

# 클래스 이름
class_names = ["Hammer", "Nipper"]

# 모델 정의 (CNN)
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 로드 함수
def load_model(model_path="custom_cnn_251002.pth"):
    model = CustomCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ResNet50 모델 로드 함수
def load_resnet50(model_path="resnet50-251010_model.pth"):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Hammer / Nipper 분류 (2개 클래스)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 모델 로드
model_cnn = load_model()
model_resnet50 = load_resnet50()

# 이미지 전처리 함수
def transform_image(image, normalize=False):
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform_list)
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# 웹페이지 렌더링
@app.route("/")
def index():
    return render_template("index-복사본-1-cnn-이미지분류기-서버사이드.html")

# 이미지 업로드 및 분류 API (CNN)
@app.route("/classify", methods=["POST"])
def classify_image():
    return classify_image_generic(model_cnn, normalize=False)

# 이미지 업로드 및 분류 API (ResNet50)
@app.route("/classify2", methods=["POST"])
def classify_image2():
    return classify_image_generic(model_resnet50, normalize=True)

# 공통 이미지 분류 함수
def classify_image_generic(model, normalize=False):
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image = Image.open(image_file).convert('RGB')
    image = transform_image(image, normalize=normalize)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

    return jsonify({
        "predicted_class": class_names[predicted_idx],
        "confidence": round(confidence * 100, 2)
    })

# Flask 서버 실행
if __name__ == "__main__":
    app.run(debug=True)
