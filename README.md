# 🌱 Plant Disease Classification using CNN and Spatial-Channel Attention [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains an approach (implemented in Python) on plant disease classification model using the a pre-trained deep learning network and fine-tuned after integration spatial and channel attention mechanisms. The based model uses VGG architecture but it can be extended to other architectures as well. We use saliency maps to show where the attention focus is on the classified images.

## 📌 Overview

The project aims to accurately classify plant diseases from images. It leverages a pre-trained model, fine-tuned with custom spatial and channel attention layers to enhance feature extraction and improve classification performance. A key feature is the use of saliency maps to visualize where the model is focusing its attention when classifying an image. **This implementation is inspired by and based on the research presented in the following article: [Handwashing Action Detection System for an Autonomous Social Robot](https://arxiv.org/pdf/2210.15804.pdf)**

## ✨ Key Features

*   **VGG16 Architecture:** Utilizes the VGG16 convolutional neural network for feature extraction. The model structure is designed to be easily adaptable to other CNN architectures.
*   **Spatial Attention:** Incorporates spatial attention mechanisms to focus on relevant regions within the input images.
*   **Channel Attention:** Implements channel attention to weigh the importance of different feature channels.
*   **Saliency Maps:** Generates saliency maps to visualize the regions of the image that the model is attending to when making its predictions. This helps in understanding the model's decision-making process.
*   **Simplicity and Extensibility:** The model is designed to be relatively simple and easy to understand, making it a good starting point for further research. The attention mechanism can be easily integrated into other CNN architectures besides VGG16.
*   **Jupyter Notebook:** The entire implementation is provided in an easy-to-follow Jupyter Notebook.  [➡️ Explore the Notebook](Plant_Disease_vgg16_spatial_channel_attention.ipynb)
*   **Transfer Learning:** Employs transfer learning to leverage pre-trained weights and accelerate training.

## ⚙️ Requirements

*   Python 3.x
*   TensorFlow
*   Keras
*   NumPy
*   Pandas
*   Scikit-learn
*   Matplotlib
*   PIL (Pillow)
*   `[requirements.txt](requirements.txt)` (see below for dependencies)

## ⬇️ Installation

1.  Clone the repository:

    ```
    git clone https://github.com/praveenpankaj/plant-disease-classification.git
    cd plant-disease-classification
    ```

2.  Install the required packages using pip:

    ```
    pip install -r requirements.txt
    ```

## 🚀 Usage

1.  Open the `[Plant_Disease_vgg16_spatial_channel_attention.ipynb](Plant_Disease_vgg16_spatial_channel_attention.ipynb)` notebook using Jupyter Notebook or JupyterLab.
2.  Follow the instructions within the notebook to train and evaluate the model. Ensure that the dataset path is correctly set within the notebook.
3.  You may need to adjust hyperparameters and dataset paths according to your specific use case.
4.  The notebook demonstrates how to generate and visualize saliency maps for classified images, providing insights into the model's attention focus.

## 📊 Dataset

*   Dataset consists of diseased plant leaf images and it's corresponding labels
*   The dataset can be directly downloaded by using the notebook from Kaggle
*   The link for the dataset is given here as reference: https://www.kaggle.com/datasets/emmarex/plantdisease (and Dataset of diseased plant leaf images and corresponding labels
*   Although the original dataset [![Full Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)](https://data.mendeley.com/datasets/tywbtsjrjv/1) contains 54303 healthy and unhealthy leaf images divided into 38 categories by species and disease, we have used only used 4352 images and 15 categoies to demonstrate the approach.
*   The images are of size 224 x 224 x 3

## 🏛️ Model Architecture

The model architecture consists of the following main components:

1.  **VGG16 Base:** A pre-trained VGG16 model (trained on ImageNet) is used as the base for feature extraction. The initial layers are typically frozen to preserve pre-trained knowledge.
2.  **Spatial Attention Module:** This module learns to attend to the most important spatial locations in the feature maps.
3.  **Channel Attention Module:** This module learns to weight the different feature channels based on their importance.
4.  **Classification Layers:** Fully connected layers followed by a softmax activation function to output the predicted class probabilities.

## 📈 Results

*  The validation accuracy was about 0.9460
* Sample Inferenced labels with ground truth labels
  ![Inferred Image Labels](https://github.com/user-attachments/assets/01354681-a3fb-454b-9e80-d2c380a26deb)


* Sample Ground Truth Images and Labels
  ![Ground Truth Images](https://github.com/user-attachments/assets/a7db8f63-a87e-43fb-a20d-99d37253da83)

and
* The corresponding Saliency Maps
 ![Salience Maps](https://github.com/user-attachments/assets/92053167-f25b-4740-84d6-d98d2d8b9897)


## Citation
* If you implement or use the above code, please don't forget to give us a shout-out and cite us! We would love to know how this work is being used.

To be cited as: S. Sasidharan et al., "Handwashing Action Detection System for an Autonomous Social Robot," TENCON 2022 - 2022 IEEE Region 10 Conference (TENCON), Hong Kong, Hong Kong, 2022, pp. 1-6, doi: 10.1109/TENCON55691.2022.9977684.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests with bug fixes, new features, or improvements to the documentation.

## 📜 License

This project is licensed under the MIT License.  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
