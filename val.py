from ultralytics import YOLO

# Load a model
model = YOLO(r'C:\Users\Leandro\temp\main\runs\classify\train2\weights\best.pt')  # load a custom model

# Validate the model
metrics = model.val(device='cpu')  # no arguments needed, dataset and settings remembered
metrics.top1   # top1 accuracy
metrics.top5   # top5 accuracy