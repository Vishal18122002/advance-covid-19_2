from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import requests

# Load the pre-trained model
model = tf.keras.models.load_model('covid_model.h5')

# Create a Flask app
app = Flask(__name__, static_url_path='/static')

# Define a route for serving the index.html file
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Preprocess the data if needed
    # For example, convert JSON data to numpy array
    features = np.array(data['features'])  # Adjust this according to your data

    # Make predictions using the loaded model
    # Assuming the model returns probabilities, where 0 represents negative and 1 represents positive
    predictions = model.predict(features)

    # Convert probabilities to predictions
    # For example, if the probability is greater than 0.5, predict positive (1), else predict negative (0)
    predicted_classes = (predictions > 0.5).astype('int32')

    # Convert predictions to human-readable labels
    # You can customize this part based on your model's output
    labels = ['Negative', 'Positive']
    predicted_labels = [labels[pred] for pred in predicted_classes]

    # Return the predicted labels as JSON
    return jsonify({'predictions': predicted_labels})

if __name__ == '__main__':
    app.run(debug=True)

# Test script to check if the model is working
# Sample data to send to the /predict route
sample_data = {
    'features': [1, 2, 3, 4]  # Adjust this according to your model's input requirements
}

# Send a POST request to the /predict route
response = requests.post('http://127.0.0.1:5000/predict', json=sample_data)

# Parse the response JSON
predicted_labels = response.json()['predictions']

# Print the predicted labels
print('Predicted Labels:', predicted_labels)
