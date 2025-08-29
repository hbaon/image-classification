#!/usr/bin/env python3
"""
AI Image Classification System
A Flask web application for real-time image classification using CNN
"""

import os
import time
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Load pre-trained ResNet50 model
print("Loading ResNet50 model...")
model = ResNet50(weights='imagenet')
print("Model loaded successfully!")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_file):
    """Preprocess image for model input"""
    try:
        # Read and resize image
        img = Image.open(image_file)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def classify_image(image_array):
    """Classify image using ResNet50 model"""
    try:
        # Make prediction
        predictions = model.predict(image_array)
        
        # Decode predictions
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        # Format results
        results = []
        for _, label, confidence in decoded_predictions:
            results.append({
                'class': label.replace('_', ' ').title(),
                'confidence': float(confidence),
                'probability': round(float(confidence) * 100, 2)
            })
        
        return results
    except Exception as e:
        print(f"Error classifying image: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """API endpoint for image classification"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, TIFF'
            }), 400
        
        # Start timing
        start_time = time.time()
        
        # Preprocess image
        img_array = preprocess_image(file)
        if img_array is None:
            return jsonify({
                'success': False,
                'error': 'Error preprocessing image'
            }), 500
        
        # Classify image
        predictions = classify_image(img_array)
        if predictions is None:
            return jsonify({
                'success': False,
                'error': 'Error classifying image'
            }), 500
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 3)
        
        # Save uploaded file (optional)
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'processing_time': processing_time,
            'filename': saved_filename
        })
        
    except Exception as e:
        print(f"Error in classify endpoint: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'ResNet50',
        'service': 'Image Classification API'
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({
        'success': False,
        'error': 'Internal server error occurred.'
    }), 500

if __name__ == '__main__':
    print("Starting AI Image Classification System...")
    print("Open your browser and navigate to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
