# app.py

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import threading
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import uuid

# ✅ Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# ✅ 모델 및 폴더 설정 (기존과 동일)
MODEL_CONFIGS = {
    "team1": {
        "model_path": "./resnet50_best_team1_animal.pth", "num_classes": 5,
        "class_labels": ["고양이", "공룡", "강아지", "꼬북이", "티벳여우"],
    },
    "team2": {
        "model_path": "./resnet50_best_team2_recycle.pth", "num_classes": 13,
        "class_labels": ["영업용_냉장고", "컴퓨터_cpu", "드럼_세탁기", "냉장고", "컴퓨터_그래픽카드", "메인보드", "전자레인지", "컴퓨터_파워", "컴퓨터_램",
                         "스탠드_에어컨", "TV", "벽걸이_에어컨", "통돌이_세탁기"],
    },
    "team3": {
        "model_path": "./resnet50_best_team3_tools_accuracy_90.pth", "num_classes": 10,
        "class_labels": ["공구 톱", "공업용가위", "그라인더", "니퍼", "드라이버", "망치", "스패너", "전동드릴", "줄자", "캘리퍼스"],
    },
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("best-busanit501-aqua.pt")
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# 🔹 YOLO 처리 함수 (안정성 강화 버전)
def process_yolo(file_path, output_path, file_type):
    try:
        if file_type == 'image':
            results = yolo_model(file_path)
            result_img = results[0].plot()
            cv2.imwrite(output_path, result_img)

        elif file_type == 'video':
            temp_output_path = output_path + ".tmp"
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"🔴 ERROR: 비디오 파일을 열 수 없습니다: {file_path}")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # FPS가 0이면 기본값 설정 (손상된 비디오 대응)
            if fps <= 0:
                fps = 30
                print(f"⚠️ WARNING: FPS를 감지할 수 없어 기본값 {fps}로 설정합니다.")

            # Windows에서 안정적인 코덱 우선순위: mp4v > XVID > MJPG
            codecs_to_try = [
                ('mp4v', '.mp4'),  # MPEG-4 (가장 호환성 좋음)
                ('XVID', '.avi'),  # Xvid (Windows 기본 지원)
                ('MJPG', '.avi'),  # Motion JPEG (폴백용)
            ]

            out = None
            for codec, ext in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    # 확장자가 다르면 임시 출력 경로 조정
                    if not temp_output_path.endswith(ext):
                        temp_output_path = os.path.splitext(temp_output_path)[0] + ext
                        output_path = os.path.splitext(output_path)[0] + ext

                    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

                    if out.isOpened():
                        print(f"✅ VideoWriter 초기화 성공: 코덱={codec}, 해상도={width}x{height}, FPS={fps}")
                        break
                    else:
                        print(f"⚠️ {codec} 코덱 실패, 다음 코덱 시도 중...")
                        out.release()
                        out = None
                except Exception as e:
                    print(f"⚠️ {codec} 코덱 오류: {e}")
                    continue

            if out is None or not out.isOpened():
                print(f"🔴 ERROR: 모든 코덱 시도 실패. OpenCV 비디오 출력을 초기화할 수 없습니다.")
                cap.release()
                return

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                results = yolo_model(frame)
                result_frame = results[0].plot()
                out.write(result_frame)

            cap.release()
            out.release()

            if os.path.exists(temp_output_path):
                os.rename(temp_output_path, output_path)
                print(f"✅ YOLO 처리 완료: {output_path}")
            else:
                print(f"🔴 ERROR: 임시 파일({temp_output_path})이 생성되지 않았습니다.")

    except Exception as e:
        print(f"🔴 ERROR in process_yolo thread: {e}")


# 🔹 기본 Index 화면
@app.route("/")
def index():
    return render_template('index.html')


# 🔹 [핵심] 예측 API (YOLO와 이미지 분류 통합)
@app.route("/predict/<model_type>", methods=["POST"])
def predict(model_type):
    if "image" not in request.files:
        return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

    file = request.files["image"]
    original_filename = file.filename
    if original_filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    # --- YOLO 모델 처리 ---
    if model_type == 'yolo':
        filename_base, file_extension = os.path.splitext(original_filename)

        if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            file_type = 'image'
        elif file_extension.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            file_type = 'video'
        else:
            return jsonify({"error": "지원되지 않는 파일 형식입니다."}), 400

        safe_filename_base = secure_filename(filename_base)
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = f"{safe_filename_base}_{unique_id}{file_extension}"

        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(file_path)

        if file_type == 'video':
            output_filename = f"result_{safe_filename_base}_{unique_id}.mp4"
        else:
            output_filename = f"result_{safe_filename}"

        output_path = os.path.join(RESULT_FOLDER, output_filename)

        thread = threading.Thread(target=process_yolo, args=(file_path, output_path, file_type))
        thread.start()

        # 전체 URL 반환 (리액트에서 직접 접근 가능)
        result_url = request.host_url.rstrip('/') + url_for('serve_result', filename=output_filename)
        status_url = request.host_url.rstrip('/') + url_for('get_status', filename=output_filename)

        return jsonify({
            "message": "YOLO 처리가 시작되었습니다.",
            "output_filename": output_filename,
            "result_url": result_url,  # 예: http://127.0.0.1:5000/results/result__6e5df66e.mp4
            "status_url": status_url  # 예: http://127.0.0.1:5000/status/result__6e5df66e.mp4
        })

    # --- 이미지 분류 모델 처리 ---
    elif model_type in MODEL_CONFIGS:
        try:
            model, class_labels = load_model(model_type)
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            # (이미지 전처리 및 예측 로직은 기존과 동일)
            transform = transforms.Compose([
                transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item() * 100

            return jsonify({
                "filename": original_filename,
                "predicted_class": class_labels[predicted_class],
                "confidence": f"{confidence:.2f}%",
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- 지원하지 않는 모델 ---
    else:
        return jsonify({"error": f"지원되지 않는 모델 유형: {model_type}"}), 400


# 🔹 처리 상태 확인 API (폴링용)
@app.route('/status/<filename>')
def get_status(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(file_path):
        return jsonify({
            "status": "complete",
            "url": url_for('serve_result', filename=filename)
        })
    else:
        return jsonify({"status": "processing"})


# 🔹 결과 파일 제공 API
@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)


# 🔹 이미지 분류 모델 로드 함수 (기존과 동일)
def load_model(model_type):
    config = MODEL_CONFIGS[model_type]
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config["num_classes"])
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.to(device)
    model.eval()
    return model, config["class_labels"]


# ✅ Flask 실행
if __name__ == "__main__":
    app.run(debug=True)