from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model, encoders, and feature order
with open('random_forest_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)

with open('feature_order.pkl', 'rb') as feature_file:
    feature_order = pickle.load(feature_file)

label_encoder_grade = encoders['TeaGrade']
label_encoder_region = encoders['Region']

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    data = request.json

    # Convert input data to DataFrame
    input_data = pd.DataFrame([{
        'TeaGrade': data['TeaGrade'],
        'Region': data['Region'],
        'AvgRainfall': data['AvgRainfall'],
        'AvgTemperature': data['AvgTemperature'],
        'QualityScore': data['QualityScore'],
        'ProductionVolume': data['ProductionVolume'],
        'CurrencyRate': data['CurrencyRate'],
        'AuctionPrice': data['AuctionPrice']
    }])

    # Encode categorical features
    input_data['TeaGradeEncoded'] = label_encoder_grade.transform(input_data['TeaGrade'])
    input_data['RegionEncoded'] = label_encoder_region.transform(input_data['Region'])

    # Drop original categorical columns
    input_data = input_data.drop(columns=['TeaGrade', 'Region'])

    # Reorder columns to match training data
    input_data = input_data[feature_order]

    # Predict demand
    prediction = rf_model.predict(input_data)[0]

    # Return prediction as JSON
    return jsonify({'PredictedAuctionDemand': prediction})

if __name__ == '__main__':
    app.run(debug=True)
