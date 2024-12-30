import io
import os
import json
import numpy as np  # Make sure to import numpy
import torch
import cv2  # Import OpenCV
from PIL import Image
from flask import Flask, jsonify, url_for, render_template, request, redirect
import argparse  # Import argparse

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Finds the model inside your directory automatically - works only if there is one model
def find_model():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    print("Please place a model file in this directory!")
    return None  # Return None if no model found

model_name = find_model()
if model_name:
    model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)
    model.eval()
else:
    raise RuntimeError("Model not found. Exiting...")  # Raise error if model not found

def get_prediction(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # Ensure image is RGB
        img_np = np.array(img)  # Convert PIL image to numpy array
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV compatibility

        # Create a writable copy of the image
        img_np.setflags(write=1)  # Make the numpy array writable

        imgs = [img_np]  # Batched list of images
        # Inference
        results = model(imgs, size=640)  # Includes NMS
        num_trees = 0
        for result in results.xyxy[0]:
            if result[-1] == 0:  # Assuming the tree class label is 0
                num_trees += 1
        return results, num_trees
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0  # Return None and 0 if error occurs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return predict()
    return render_template('index.html')

def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files.get('file')
    if not file:
        return redirect(request.url)  # Redirect if no file is found
        
    img_bytes = file.read()
    results, num_trees = get_prediction(img_bytes)
    if results is None:
        return "Error processing image", 500  # Return error if processing fails
    
    results.save(save_dir='static')  # Save results to static folder
    filename = 'image0.jpg'  # Ensure this matches the output
    
    return render_template('result.html', result_image=filename, model_name=model_name, num_trees=num_trees)

@app.route('/detect', methods=['GET', 'POST'])
def handle_video():
    # Some code to be implemented later
    pass

@app.route('/webcam', methods=['GET', 'POST'])
def web_cam():
    # Some code to be implemented later
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv7 models")
    parser.add_argument("--port", default=5000, type=int, help="Port number")
    args = parser.parse_args()
    
    app.run(host="0.0.0.0", port=args.port)