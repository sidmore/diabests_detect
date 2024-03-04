from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained models (ensure the models are in the same directory as this script)
modelA = joblib.load("diabetes_modelA.pkl")
modelB = joblib.load("diabetes_modelB.pkl")

models = {
    'modelA': modelA,
    'modelB': modelB
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract features and model choice from the request
    features = [data['features']]
    model_name = data.get('model_name', 'modelA')  # Default to modelA if not specified
    
    # Select the model based on the input parameter
    if model_name in models:
        selected_model = models[model_name]
    else:
        return jsonify({'error': 'Model not found'}), 404
    
    # Make prediction
    prediction = selected_model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)




# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# import numpy as np
# import joblib

# app = Flask(__name__)

# # Load the model and scaler
# model = load_model('model/diabetes_model.h5')
# scaler = joblib.load('model/scaler.pkl')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     features = np.array(data['features'])
    
#     # Scale the features
#     features_scaled = scaler.transform([features])
    
#     # Make prediction
#     prediction = model.predict(features_scaled)
#     prediction_label = (prediction > 0.5).astype(int)
    
#     return jsonify({'prediction': prediction.tolist(), 'prediction_label': prediction_label.tolist()})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

