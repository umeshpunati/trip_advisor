from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model.pkl')
trip_encoder = joblib.load('trip_encoder.pkl')
transport_encoder = joblib.load('transport_encoder.pkl')
destination_encoder = joblib.load('destination_encoder.pkl')

# Define valid options
valid_trip_types = ['Adventure', 'Relaxation', 'Cultural', 'Honeymoon', 'Historical', 'Hill Station', 'Beach', 'Nature']
valid_transport_modes = ['Road', 'Flight']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Validate required fields
        if 'budget' not in request.form or 'trip_type' not in request.form or 'mode_transport' not in request.form:
            return jsonify({'error': 'Missing required fields'}), 400

        # Get form data
        budget = float(request.form['budget'])
        trip_type = request.form['trip_type']
        mode_transport = request.form['mode_transport']

        # Validate inputs
        if budget < 2000:
            return jsonify({'error': 'Budget too low. Minimum budget is 2000.'}), 400

        if trip_type not in valid_trip_types:
            return jsonify({'error': f'Invalid trip type. Please choose from {valid_trip_types}.'}), 400

        if mode_transport not in valid_transport_modes:
            return jsonify({'error': f'Invalid transport mode. Please choose from {valid_transport_modes}.'}), 400

        # Encode inputs
        trip_encoded = trip_encoder.transform([trip_type])[0]
        transport_encoded = transport_encoder.transform([mode_transport])[0]

        # Predict
        input_data = np.array([[budget, trip_encoded, transport_encoded]])
        prediction = model.predict(input_data)
        destination = destination_encoder.inverse_transform(prediction)[0]

        return jsonify({'destination': destination})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Disable debug mode in production