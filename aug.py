import os
from PIL import Image

# Set the input and output directories
input_dir = r"C:\Users\Leandro\temp\main\data\new_data_plaque"
output_dir = r"C:\Users\Leandro\temp\main\data\images\flip"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the function to flip the image
def flip_image(image, mode):
    return image.transpose(mode)

# Define the modes for flipping
flip_modes = {
    "FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
    "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
}

# Iterate over the input directory and subdirectories
for root, dirs, files in os.walk(input_dir):
    class_name = os.path.basename(root)
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(root, file)
            image = Image.open(image_path).convert("RGB")  # Convert to RGB mode

            # Save the original image
            output_path = os.path.join(class_output_dir, file)
            image.save(output_path)

            # Flip the image and save it
            for mode_name, mode in flip_modes.items():
                flipped_image = flip_image(image, mode)
                base_name = os.path.splitext(file)[0]
                new_file_name = f"{base_name}_{mode_name}.png"
                output_path = os.path.join(class_output_dir, new_file_name)
                flipped_image.save(output_path)