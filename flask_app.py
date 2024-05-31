from flask import Flask, request, jsonify
#pip install ultralytics==8.0.230 
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)

# Load the YOLO model
model = YOLO(r'C:\Users\Leandro\temp\main\runs\classify\derma_model_n\weights\last.pt')
class_names = model.model.names

# Create a dictionary to map class indices to custom class names
custom_class_names = {0: 'ATOPIC DERMATITIS', 1: 'PLAQUE PSORIASIS'}

@app.route('/classify', methods=['POST'])
def classify_image():
    # Check if the POST request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    # Check if the file is an image
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the image to 128x128 using bicubic interpolation
    img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)

    # Run inference
    results = model.predict(img_resized, save=False, imgsz=128, conf=0.90)

    threshold = 0.90
    response = {}

    for result in results:
        score = result.probs.top1conf
        prediction = result.probs.data.cpu().numpy()

        if score < threshold:
            response['result'] = "Can't classify other skin disease.\n NOTE: Please consult a dermatologist for a professional diagnosis regardless of the result."
        else:
            top_class_idx = prediction.argmax()
            class_name = custom_class_names.get(top_class_idx, class_names[top_class_idx])
            response['result'] = f"Classified as {class_name} with a confidence score of {score * 100:.2f}%.\nNOTE: Please consult a dermatologist for a professional diagnosis regardless of the result."

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")