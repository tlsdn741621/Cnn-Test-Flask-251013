from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# --- 1. 애플리케이션 및 모델/스케일러 초기화 ---

app = Flask(__name__)


# 각 모델의 클래스 정의 (파일 상단에 모두 정의)
class StockPredictorRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, output_size=1):
        super(StockPredictorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=1, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc(self.relu(last_out))
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, output_size=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1])
        return out


# 모델과 스케일러를 담을 딕셔너리 생성
models = {}
scalers = {}
device = torch.device('cpu')

try:
    # RNN 모델 및 스케일러 로드
    models['RNN'] = StockPredictorRNN()
    models['RNN'].load_state_dict(torch.load('Rnn-samsungStock.pth', map_location=device))
    models['RNN'].eval()
    scalers['RNN'] = torch.load('Rnn-scaler.pth', map_location=device, weights_only=False)

    # LSTM 모델 및 스케일러 로드
    models['LSTM'] = LSTMModel()
    models['LSTM'].load_state_dict(torch.load('samsungStock_LSTM_60days_basic.pth', map_location=device))
    models['LSTM'].eval()
    scalers['LSTM'] = torch.load('scaler_LSTM_60days_basic.pth', map_location=device, weights_only=False)

    # GRU 모델 및 스케일러 로드
    models['GRU'] = GRUModel()
    models['GRU'].load_state_dict(torch.load('samsungStock_GRU.pth', map_location=device))
    models['GRU'].eval()
    scalers['GRU'] = torch.load('scaler_GRU.pth', map_location=device, weights_only=False)

except FileNotFoundError as e:
    print(f"오류: 모델 또는 스케일러 파일을 찾을 수 없습니다. ({e.filename})")
    print("모든 .pth 파일이 app.py와 동일한 폴더에 있는지 확인해주세요.")

# CSV 데이터 로드
try:
    stock_df = pd.read_csv('7-samsung_stock_2022_01_2025_10_13.csv', index_col='Date', parse_dates=True)
    stock_df.sort_index(inplace=True)  # 날짜순으로 정렬
except FileNotFoundError:
    print("오류: 'samsung_3year_price.csv' 파일을 찾을 수 없습니다.")
    stock_df = None


# --- 2. Flask 라우트(API 엔드포인트) 정의 ---

@app.route('/')
def home():
    """메인 페이지 (index.html)를 렌더링합니다."""
    return render_template('index.html')


@app.route('/api/stockdata')
def get_stockdata():
    """CSV 파일에서 주가 데이터를 가져와 JSON으로 응답합니다."""
    if stock_df is None:
        return jsonify({"error": "서버에 데이터 파일이 없습니다."}), 500

    period = request.args.get('period', '5d')

    # 컬럼명을 프론트엔드와 맞추기 위해 첫 글자 대문자로 변경
    df_copy = stock_df.copy()
    df_copy.columns = [col.capitalize() for col in df_copy.columns]

    period_map = {'1d': 1, '5d': 5}  # 예측에 필요한 일 수
    days = period_map.get(period, 5)

    # 가장 최근 데이터부터 'days'만큼 가져오기
    recent_data = df_copy.tail(days)

    # 예측에 필요한 과거 데이터 선택
    if period == '1d':
        # 1일 예측 시, 가장 최근 1일치 데이터 사용
        data_subset = recent_data
    else:  # '5d' 등 다른 기간
        # 다음 날 예측을 위해 가장 최근 하루를 제외한 데이터를 사용
        data_subset = recent_data.iloc[:-1]

    # JSON으로 변환
    data_subset = data_subset.reset_index()
    data_subset['Date'] = data_subset['Date'].dt.strftime('%Y-%m-%d')
    stock_data_json = data_subset.to_dict(orient='records')

    return jsonify(stock_data_json)


@app.route('/api/predict', methods=['POST'])
def predict():
    """하나의 엔드포인트에서 모든 모델의 예측을 처리합니다."""
    try:
        req_data = request.get_json()
        model_type = req_data.get('model')  # 'RNN', 'LSTM', 'GRU'
        input_data = req_data.get('data')

        if not all([model_type, input_data]):
            return jsonify({"error": "모델 타입 또는 데이터가 없습니다."}), 400

        # 요청된 모델 타입에 맞는 모델과 스케일러 선택
        model = models.get(model_type)
        scaler = scalers.get(model_type)

        if not model or not scaler:
            return jsonify({"error": f"'{model_type}' 모델을 찾을 수 없습니다."}), 404

        # 예측 로직
        input_np = np.array(input_data)
        input_scaled = scaler.transform(input_np)
        input_tensor = torch.Tensor(input_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction_scaled = model(input_tensor).item()

        # 원래 값으로 복원
        prediction = scaler.inverse_transform([[0, 0, 0, prediction_scaled]])[0][3]

        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": "예측 중 오류 발생", "details": str(e)}), 500

#레스트용,
@app.route('/api/predict2/<string:model_type>', methods=['POST'])
def predict2(model_type):
    """
    URL 경로에서 모델 타입을 받고, JSON 본문에서 'data'와 'period'를 받아 예측합니다.
    """
    try:
        req_data = request.get_json()
        input_data = req_data.get('data')
        period = req_data.get('period')  # period도 검증을 위해 받습니다.

        # 1. 필수 데이터 검증
        if not all([input_data, period]):
            return jsonify({"error": "요청 본문에 data와 period 정보가 모두 포함되어야 합니다."}), 400

        # 2. 요청된 모델과 스케일러 선택 (URL에서 받은 model_type 사용)
        model = models.get(model_type.upper())  # 대소문자 구분 없도록 .upper()
        scaler = scalers.get(model_type.upper())

        if not model or not scaler:
            return jsonify({"error": f"'{model_type}' 모델을 서버에서 찾을 수 없습니다."}), 404

        # 3. 'period' 값을 사용해 'data'의 길이 검증
        period_days_map = {'1d': 1, '5d': 4}
        expected_length = period_days_map.get(period)

        if expected_length is None:
            return jsonify({"error": f"지원되지 않는 기간입니다: {period}"}), 400

        if len(input_data) != expected_length:
            return jsonify({"error": f"데이터 길이가 잘못되었습니다. '{period}' 기간에는 {expected_length}일치 데이터가 필요합니다."}), 400

        # 4. 예측 로직
        input_np = np.array(input_data)
        input_scaled = scaler.transform(input_np)
        input_tensor = torch.Tensor(input_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction_scaled = model(input_tensor).item()

        prediction = scaler.inverse_transform([[0, 0, 0, prediction_scaled]])[0][3]

        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": "예측 중 서버 오류가 발생했습니다.", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)