from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

# Load the saved model
model = tf.keras.models.load_model('tuned_handwritten_digit_model.h5')

# Preprocess the uploaded image on the server
def preprocess_image(image):
    # Convert the uploaded image to grayscale and resize it to 28x28
    img = image.convert('L')
    img = img.resize((28, 28))
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    
    # Reshape the array to match the model's expected input shape (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/')
def home():
    return "Hello from the Handwritten Digit Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))

        # Check if the image is a valid format
        if image is None or image.format not in ['PNG', 'JPEG']:
            return jsonify({'error': 'Invalid image format'}), 400

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make a prediction using the model
        prediction = model.predict(processed_image)
        prediction_class = np.argmax(prediction, axis=1)[0]

        # Return the prediction as a JSON response
        return jsonify({'prediction': int(prediction_class)})

    except Exception as e:
        # Handle any exception during processing
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
