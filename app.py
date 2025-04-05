import io
import os
import numpy as np
import torch
import cv2
from PIL import Image
from flask import Flask, jsonify, url_for, render_template, request, redirect
import argparse

app = Flask(__name__)
RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Find the model file (.pt) in the current directory
def find_model():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    print("Please place a model file in this directory!")
    return None

model_name = find_model()
if model_name:
    model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name)
    model.eval()
else:
    raise RuntimeError("Model not found. Exiting...")

def get_prediction(img_bytes):
    try:
        # Load the image and prepare it for inference
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Perform inference using the YOLOv7 model
        results = model([img_np], size=640)

        # Initialize variables for calculating areas and tracking trees
        img_height, img_width, _ = img_np.shape
        total_img_area = img_height * img_width
        total_tree_area = 0
        tree_boxes = []
        health_stats = {
            "Healthy": 0,
            "Moderate": 0,
            "Unhealthy": 0,
            "Uncertain": 0
        }

        # Process detection results and classify each tree's health status
        for result in results.xyxy[0]:
            if result[-1] == 0:  # Assuming tree class label is 0
                x1, y1, x2, y2 = map(int, result[:4])
                area = (x2 - x1) * (y2 - y1)
                total_tree_area += area
                tree_boxes.append((x1, y1, x2, y2))

                # Extract sub-image and classify health status based on crown color
                tree_crop = img_np[y1:y2, x1:x2]
                health_status = classify_health_by_color(tree_crop)
                health_stats[health_status] += 1

                # Draw bounding box with color based on health status
                color = get_color(health_status)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img_np, health_status, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        # Calculate the total tree area as a percentage of the image area
        tree_area_percentage = round((total_tree_area / total_img_area) * 100, 2)

        # Save the result image
        result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
        cv2.imwrite(result_image_path, img_np)

        return len(tree_boxes), tree_area_percentage, 'result.jpg', health_stats

    except Exception as e:
        print(f"Error during prediction: {e}")
        return 0, 0, None, None

def classify_health_by_color(image):
    """Classify tree health based on average color of the crown."""
    # Calculate the average color of the crown (in BGR format)
    avg_color = np.mean(image, axis=(0, 1))
    blue, green, red = avg_color

    # Determine health based on crown color characteristics
    if green > red and green > blue:
        return "Healthy"  # Green crown
    elif red > green and green > blue:  
        return "Moderate"  # Yellow crown (red + green dominant)
    elif np.mean(avg_color) < 100:  # Dark crown (low brightness)
        return "Unhealthy"  # Dark/Brown crown
    else:
        return "Uncertain"  # Default fallback

def get_color(health_status):
    """Get color based on the health status."""
    if health_status == "Healthy":
        return (0, 255, 0)  # Green
    elif health_status == "Moderate":
        return (0, 255, 255)  # Yellow
    elif health_status == "Unhealthy":
        return (0, 0, 255)  # Red
    else:
        return (255, 255, 255)  # White (for uncertain cases)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            img_bytes = file.read()
            num_trees, tree_area_percentage, result_image, health_stats = get_prediction(img_bytes)
            
            if not result_image:
                return jsonify({'error': 'Image processing failed'}), 500
                
            return jsonify({
                'success': True,
                'result_image': result_image,
                'num_trees': num_trees,
                'tree_area_percentage': tree_area_percentage,
                'health_stats': health_stats
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv7 models")
    parser.add_argument("--port", default=5000, type=int, help="Port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
