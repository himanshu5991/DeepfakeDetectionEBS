import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from custom_layers import CustomDepthwiseConv2D

# Initialize the Flask application
app = Flask(__name__)

# Define the path to your saved model
model_path = 'final_model.h5'

# Load the model using the custom layer
model = tf.keras.models.load_model(model_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

# Define the allowed extensions for video files
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    total_prediction_value = 0.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess the frame as required
        frame_resized = cv2.resize(frame, (224, 224))
        frame_array = np.expand_dims(frame_resized, axis=0)
        # Make prediction
        prediction = model.predict(frame_array)
        prediction_value = float(prediction[0][0])
        total_prediction_value += prediction_value
        frame_number += 1
    cap.release()

    average_prediction_value = total_prediction_value / frame_number if frame_number > 0 else 0
    threshold = 0.5  # Example threshold value
    video_classification = "fake" if average_prediction_value > threshold else "real"

    return video_classification, average_prediction_value

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the video file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)
        video_classification, average_prediction_value = process_video(file_path)
        os.remove(file_path)  # Clean up the saved file after processing
        return jsonify({
            'classification': video_classification,
            'score': average_prediction_value
        }), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
