# Thesis Repository: Using GAN and YOLOv8 for Classification

Welcome to my thesis repository for my college years at Bicol University. This repository contains all the code, data, and documentation related to my thesis work, which focuses on utilizing Generative Adversarial Networks (GANs) and YOLOv8 for image classification tasks.

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This thesis explores the application of Generative Adversarial Networks (GANs) to augment training datasets and improve the performance of image classification models. After generating synthetic images using GANs, YOLOv8, a state-of-the-art object detection and classification model, is used to classify the images. The goal is to enhance the accuracy and robustness of the classification process by leveraging the capabilities of GANs for data augmentation.

## Technologies Used

- **Python**: Programming language
- **TensorFlow/Keras**: For building and training GANs
- **YOLOv8**: For image classification
- **OpenCV**: For image processing
- **NumPy**: For numerical operations
- **Matplotlib**: For data visualization

## Installation

To run the code in this repository, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/thesis-repo.git
    cd thesis-repo
    ```

2. Create a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Training GAN**: Use the scripts in the `gan` directory to train your GAN model. You can modify the parameters in the configuration file to fit your specific requirements.

    ```sh
    python gan/train_gan.py --config gan/config.yaml
    ```

2. **Generating Images**: After training the GAN, generate synthetic images using the trained model.

    ```sh
    python gan/generate_images.py --model gan/model.h5 --output generated_images/
    ```

3. **Classification with YOLOv8**: Use the YOLOv8 scripts to classify images, including both original and generated images.

    ```sh
    python yolov8/classify.py --images dataset/images/ --output results/
    ```

## Project Structure

