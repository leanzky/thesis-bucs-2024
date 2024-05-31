import os
import shutil
from sklearn.model_selection import train_test_split, StratifiedKFold
from glob import glob
import numpy as np
from ultralytics import YOLO

# Define paths
base_path = r'C:\Users\Leandro\temp\main\train\data\images'
mix_no_split_path = r'C:\Users\Leandro\temp\main\train\data\mix_no_split'
folds_path = base_path

# Get image paths and labels
image_paths = []
labels = []

for class_name in os.listdir(mix_no_split_path):
    class_path = os.path.join(mix_no_split_path, class_name)
    if not os.path.isdir(class_path):
        continue
    # Look for .png and .jpg files
    for image_path in glob(os.path.join(class_path, '*.png')) + glob(os.path.join(class_path, '*.jpg')):
        image_paths.append(image_path)
        labels.append(class_name)

# Check if any images and labels were found
if len(image_paths) == 0 or len(labels) == 0:
    raise ValueError("No images found in the dataset. Please check your dataset directory structure.")

# Convert lists to numpy arrays
image_paths = np.array(image_paths)
labels = np.array(labels)

# Print some debug information
print(f"Found {len(image_paths)} images.")
print(f"Classes found: {set(labels)}")

# Split into train and validation sets (80-20 split)
train_index, val_index = train_test_split(
    np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42)

# Define function to create directory if not exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Create initial train/val split directories
train_path = os.path.join(base_path, 'train')
val_path = os.path.join(base_path, 'val')
for class_name in set(labels):
    ensure_dir(os.path.join(train_path, class_name))
    ensure_dir(os.path.join(val_path, class_name))

# Copy training images
for idx in train_index:
    src = image_paths[idx]
    dst = src.replace(mix_no_split_path, train_path)
    ensure_dir(os.path.dirname(dst))
    shutil.copy(src, dst)

# Copy validation images
for idx in val_index:
    src = image_paths[idx]
    dst = src.replace(mix_no_split_path, val_path)
    ensure_dir(os.path.dirname(dst))
    shutil.copy(src, dst)

# Now perform k-fold cross-validation on the training set
train_image_paths = image_paths[train_index]
train_labels = labels[train_index]

# Define the number of folds
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Start k-fold cross-validation
for fold, (train_fold_index, val_fold_index) in enumerate(skf.split(train_image_paths, train_labels)):
    print(f'Fold {fold + 1}')
    
    # Create fold-specific directories
    fold_base_path = os.path.join(folds_path, f'fold_{fold+1}')
    fold_train_path = os.path.join(fold_base_path, 'train')
    fold_val_path = os.path.join(fold_base_path, 'val')
    
    for class_name in set(labels):
        ensure_dir(os.path.join(fold_train_path, class_name))
        ensure_dir(os.path.join(fold_val_path, class_name))
    
    # Copy fold training images
    for idx in train_fold_index:
        src = train_image_paths[idx]
        dst = src.replace(mix_no_split_path, fold_train_path)
        ensure_dir(os.path.dirname(dst))
        shutil.copy(src, dst)
    
    # Copy fold validation images
    for idx in val_fold_index:
        src = train_image_paths[idx]
        dst = src.replace(mix_no_split_path, fold_val_path)
        ensure_dir(os.path.dirname(dst))
        shutil.copy(src, dst)
    
    # Initialize and train YOLO model with specified parameters
    model = YOLO('yolov8n-cls.pt')
    results = model.train(data=fold_base_path, epochs=10, imgsz=128, task='classify', device='cpu', batch=16)

    # Evaluate model
    val_results = model.val()
    print(f'Fold {fold + 1} validation results: {val_results}')

# Clean up the fold directories if needed
# shutil.rmtree(os.path.join(base_path, 'fold_*'))