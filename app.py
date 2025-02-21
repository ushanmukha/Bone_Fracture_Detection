from flask import Flask, render_template, request, jsonify
import torch
import cv2
import base64
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the YOLOv11 model
model = YOLO('models/best2155e.pt')

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run detection with YOLOv11
        results = model(filepath)
        
        if len(results[0].boxes) == 0:
            result = "No fractures detected"
            os.remove(filepath)
            return jsonify({"result": result})
        else:
            # Get the image with annotations
            rendered_img = results[0].plot()
            rendered_img_rgb = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', rendered_img_rgb)
            img_str = base64.b64encode(buffer).decode('utf-8')
            os.remove(filepath)
            return jsonify({"result": "Fractures detected", "image": img_str})
    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER']) 
    app.run(debug=True)
