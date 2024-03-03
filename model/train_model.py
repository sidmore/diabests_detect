import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, names=column_names)

# Split the data into features and target
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Keras model
model = Sequential([
    Dense(12, input_shape=(8,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=1)

# Save the model and scaler
model.save('diabetes_model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved to disk.")

# Function to make predictions
def predict_diabetes(input_features):
    # Load the model and scaler from disk
    model = load_model('diabetes_model.h5')
    scaler = joblib.load('scaler.pkl')
    
    # Scale the input features
    input_features_scaled = scaler.transform([input_features])
    
    # Make prediction
    prediction = model.predict(input_features_scaled)
    
    # Convert prediction probability to class label
    prediction_label = (prediction > 0.5).astype(int)
    
    return prediction[0][0], prediction_label[0][0]

# Example usage:
input_features = np.array([6, 148, 72, 35, 0, 33.6, 0.627, 50])
prediction_probability, prediction_label = predict_diabetes(input_features)
print(f"Prediction probability: {prediction_probability}")
print(f"Prediction label (1 for Diabetes, 0 for No Diabetes): {prediction_label}")

