from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the saved models and encoders
gbr = joblib.load('gbr_tea_yield_model.pkl')
label_encoder = joblib.load('label_encoder_month.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have a corresponding index.html file

# Define route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check for JSON input
        data = request.json
        print("Received JSON data:", data)

        # Extract input values from JSON
        soil_type = data['soil_type']
        soil_ph = float(data['soil_ph'])
        humidity = float(data['humidity'])
        temperature = float(data['temperature'])
        sunlight_hours = float(data['sunlight_hours'])
        month = data['month']
        plant_age = float(data['plant_age'])
        rainfall = float(data['rainfall'])

        # Prepare the input data
        input_data = pd.DataFrame([[soil_type, soil_ph, humidity, temperature, sunlight_hours, month, plant_age, rainfall]],
                                  columns=['Soil type', 'Soil pH', 'Humidity (%)', 'Temperature (°C)', 'Sunlight Hours', 'Month', 'Plant Age (years)', 'Rainfall (mm)'])

        # Encode and preprocess
        input_data['Month'] = label_encoder.transform(input_data['Month'])
        input_data = pd.get_dummies(input_data, columns=['Soil type'], drop_first=True)

        # Load training columns and add missing columns
        train_columns = joblib.load('train_columns.pkl')
        missing_cols = set(train_columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[train_columns]

        # Impute and scale
        input_data_imputed = imputer.transform(input_data)
        input_data_scaled = scaler.transform(input_data_imputed)

        # Predict
        prediction = gbr.predict(input_data_scaled)

        return jsonify({'Predicted Yield (kg/ha)': f'{prediction[0]:.2f}'})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
