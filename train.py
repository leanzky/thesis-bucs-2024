import os
from ultralytics import YOLO
import torch

os.environ['WANDB_DISABLED'] = 'true'

# Load a model
#model = YOLO(r"C:\Users\Leandro\temp\main\ultralytics\ultralytics\cfg\models\v8\yolov8-cls.yaml")  # build a new model from YAML
model = YOLO("yolov8n-cls.pt")
print(torch.cuda.is_available())

# Train the model
results = model.train(data=r'C:\Users\Leandro\temp\main\train\images', 
                      epochs=20,
                      imgsz=128, 
                      task='classify', 
                      device="cuda:0", 
                      batch=16,
                      val=True,
                      dropout=0.3,
                      name='fish_model',
                      verbose=True,
                      cls=0.5,
                      plots=True,
                      #Augmentation
                      flipud=1,
                      fliplr=1,
                      hsv_h=0.015,
                      hsv_s=0.5,
                      scale=0.7,
                      mixup=0.5,
                      patience=100,
                      degrees=180,
                      translate=0.35,
                      perspective=0.01,
                      mosaic=0.5,
                      label_smoothing=0.1,
                      workers=0,
                      shear=180,
                      close_mosaic=5
                      )