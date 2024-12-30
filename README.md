# Tree Enumeration and Health Analysis System  

This project provides an advanced solution for analyzing satellite or drone images to count trees, detect their locations, and assess their health based on the color of the tree crown. Leveraging the YOLOv7 model for object detection, it offers precise and efficient tree analysis.

---

## Features  
- **Upload Images:** Accepts satellite or drone images in various formats (e.g., JPG, PNG).  
- **Tree Detection:** Identifies and counts trees in the uploaded image.  
- **Bounding Boxes:** Draws bounding boxes around detected trees.  
- **Tree Health Analysis:** Evaluates tree health based on the crown color and categorizes it.  
- **User-Friendly Interface:** Simple and interactive frontend for uploading images and viewing results.  

---

## Tech Stack  
- **Backend:** YOLOv7, Flask  
- **Frontend:** HTML, CSS, JavaScript (React.js integration optional)  
- **Libraries:** OpenCV, NumPy, PyTorch  

---

## Installation and Setup  

### 1. Clone the Repository  
```bash  
git clone https://github.com/your-username/Tree_Enumeration_YoloV7.git  
cd Tree_Enumeration_YoloV7  
2. Set Up the Environment
Install Python (>= 3.8) and pip.
Create and activate a virtual environment:
bash
Copy code
python -m venv env  
source env/bin/activate  # On Windows: env\Scripts\activate  
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt  
4. Download YOLOv7 Weights
Place the best.pt YOLOv7 weights file in the models/ directory.
5. Run the Application
bash
Copy code
python app.py  
Access the application in your browser at http://127.0.0.1:5000.

Usage
Upload Image: Select a satellite or drone image to upload.
Process Image: The system detects trees, counts them, and analyzes their health.
View Results: See the output with bounding boxes, tree count, and health status.
Directory Structure
php
Copy code
Tree_Enumeration_YoloV7/  
├── models/                 # YOLOv7 weights (e.g., best.pt)  
├── static/                 # Frontend assets (CSS, JS)  
├── templates/              # HTML templates for Flask  
├── app.py                  # Flask application  
├── requirements.txt        # Python dependencies  
└── README.md               # Project documentation  
Future Enhancements
Add React.js frontend for better interactivity.
Support real-time image analysis through video feeds.
Integrate map-based visualization for geolocation of trees.
Contributing
Feel free to open issues or create pull requests for improvements!



https://github.com/user-attachments/assets/bc86226e-403c-4954-b143-319ec5f2e1a7

