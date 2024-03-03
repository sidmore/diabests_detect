from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
model = load_model('model/diabetes_model.h5')
scaler = joblib.load('model/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features'])
    
    # Scale the features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)
    prediction_label = (prediction > 0.5).astype(int)
    
    return jsonify({'prediction': prediction.tolist(), 'prediction_label': prediction_label.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

