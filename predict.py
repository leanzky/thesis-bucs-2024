from ultralytics import YOLO
import numpy
import cv2

# Load a pretrained YOLOv8n model
model = YOLO(r'C:\Users\Leandro\temp\main\runs\classify\fish_model\weights\last.pt')

# Read the image
img = cv2.imread('C:\\Users\\Leandro\\temp\\main\\test\\redtail.jpg')

# Run inference on 'bus.jpg' with arguments
results = model.predict(img, save=False, imgsz=128)
threshold = 0.90
# Get the class names from the model
class_names = model.model.names

for result in results:
    for result in results:
        score = result.probs.top1conf
        prediction = result.probs.data.cpu().numpy()
    if score < threshold:
        print(f"Can't classify")
    else:
        top_class_idx = prediction.argmax()
        class_name = class_names[top_class_idx]
        print(f"Classified as {class_name} with a confidence score of {score * 100:.2f}%")