import requests

# Replace the file path with the actual path to your test image
image_path = 'C:\\Users\\Leandro\\temp\\main\\test\\leg.jpg'

with open(image_path, 'rb') as f:
    image_data = f.read()

response = requests.post('http://192.168.1.6:5000/classify', files={'image': image_data})
print(response.text)