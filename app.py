from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained models (ensure the models are in the same directory as this script)
model_a = load_model('model/diabetes_model1.h5')
scaler_a = joblib.load('model/scaler1.pkl')

model_b = load_model('model/diabetes_model2.h5')
scaler_b = joblib.load('model/scaler2.pkl')

models = {
    'a': model_a,
    'b': model_b
}

scalers = {
    'a': scaler_a,
    'b': scaler_b
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract features and model choice from the request
    features = [data['features']]
    model_name = data.get('model_name', 'a')  # Default to model1 if not specified
    
    # Select the model based on the input parameter
    if model_name in models:
        selected_model = models[model_name]
        selected_scaler = scalers[model_name]
    else:
        return jsonify({'error': 'Model not found'}), 404
    
    # Make prediction
    features_scaled = selected_scaler.transform([features])
    prediction = selected_model.predict(features_scaled)
    prediction_label = (prediction > 0.5).astype(int)

    return jsonify({'prediction': prediction.tolist(), 'prediction_label': prediction_label.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
