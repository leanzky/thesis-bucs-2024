# Thesis Repository: Using GAN and YOLOv8 for Classification

Welcome to my thesis repository for my college years at Bicol University. This repository contains all the code, data, and documentation related to my thesis work, which focuses on utilizing Generative Adversarial Networks (GANs) and YOLOv8 for image classification tasks.

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Introduction

This thesis explores the application of Generative Adversarial Networks (GANs) to augment training datasets and improve the performance of image classification models. After generating synthetic images using GANs, YOLOv8, a state-of-the-art object detection and classification model, is used to classify the images. The goal is to enhance the accuracy and robustness of the classification process by leveraging the capabilities of GANs for data augmentation.

## Technologies Used

- **Python**: 3.8+
- **GAN Repository(Credits)**: [For building and training GANs](https://github.com/odegeasslbc/FastGAN-pytorch.git)
- **YOLOv8**: 8.2.7

## Installation

To run the code in this repository, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/leanzky/thesis-bucs-2024.git
    cd thesis-bucs-2024
    ```
2. Create a virtual environment (use miniconda3).
3. Install the required dependencies.

## Usage

1. **Training GAN**: Use the script in fsgan.ipynb in Google Collab.
2. **Classification with YOLOv8**: Use the predict.py and configure everything base on the comments.

## Acknowledgements

I would like to thank the following:

- **Leandro Francia (Me)**
- **Charlene Cortes**
- **Kiabelle Bilan**

Additionally, I would like to acknowledge the repository [FastGAN-pytorch](https://github.com/odegeasslbc/FastGAN-pytorch?tab=readme-ov-file) for providing an excellent foundation and resources that were instrumental in the development of this thesis.

I would also like to thank my advisor, professors, and colleagues at Bicol University for their support and guidance throughout this project. Special thanks to the open-source community for providing the tools and frameworks that made this work possible.

