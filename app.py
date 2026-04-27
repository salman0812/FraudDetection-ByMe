from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load('model_fraud.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/fraud', methods=['POST'])
def predict_fraud():
    try:
        data = request.get_json()
        features = [data.get(f, 0) for f in feature_names]
        X = np.array(features).reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        prediction = int(prob >= 0.5)

        if prob < 0.4:
            risk_level = 'LOW'
        elif prob < 0.7:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'

        return jsonify({
            'prediction': prediction,
            'probability': float(prob),
            'risk_level': risk_level,
            'alert': bool(prediction == 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({
        'model': {
            'type': 'XGBoost',
            'auc_roc': 0.9760,
            'precision': 0.88,
            'recall': 0.84,
            'f1_score': 0.86,
            'dataset_size': 284807
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
