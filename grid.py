import os
import random
from PIL import Image

# Folder path
folder_path = r"C:\Users\Leandro\temp\main\data\bicubic_atopic_128"

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle the list of image files randomly
random.shuffle(image_files)

# Open the first 25 images
images = [Image.open(os.path.join(folder_path, f)).resize((128, 128)) for f in image_files[:25]]

# Create a new image with the size of the grid
grid_width = 3 * images[0].width
grid_height = 3 * images[0].height
grid = Image.new('RGB', (grid_width, grid_height))

# Paste the images into the grid
for i in range(5):
    for j in range(5):
        index = i * 5 + j
        grid.paste(images[index], (j * images[index].width, i * images[index].height))

# Save the grid as a PNG file
grid.save('image_grid_atopic.png')
print("Image grid saved as 'image_grid_plaque.png'")