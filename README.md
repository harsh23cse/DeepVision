# DeepVision — Interactive Learning of AI Models


DeepVision is an interactive, browser-based educational platform designed to visualize and demystify the inner workings of deep learning models. By running real-time inference using pre-trained TensorFlow.js models directly in the browser, DeepVision allows users to visually explore how different neural network architectures process data, update weights, and make predictions layer by layer.

## 🚀 Features & Supported Models

The dashboard provides deep-dive visualizations for four core architectures:

*   **CNN (Convolutional Neural Network - ResNet):** Explore how deep networks learn visual features from images. Upload custom images to see real-time convolution scanning, ReLU activation, batch normalization, and max pooling in action.
*   **ANN (Artificial Neural Network):** Watch how weights update and loss reduces step by step during the training and inference processes.
*   **GAN (Generative Adversarial Network):** Visualize the adversarial competition between a Generator and Discriminator as the AI learns to generate realistic data.
*   **YOLO (You Only Look Once):** Understand how AI detects and localizes objects in real-time, visualizing bounding box confidence scores and grid-based scanning.

## 🛠️ Technology Stack

*   **Frontend:** HTML5, Vanilla JavaScript, CSS3
*   **Build Tool:** Vite
*   **Machine Learning:** TensorFlow.js (TFJS)
*   **Pre-trained Models Used:**
    *   [MobileNet / ResNet](https://www.image-net.org/) (ImageNet Dataset)
    *   [YOLO](https://cocodataset.org/) (COCO Dataset)
    *   [MNIST](http://yann.lecun.com/exdb/mnist/) (Basic ANN/CNN demonstrations)

## 💻 Running the Project Locally

Because the project uses modern ES modules and Vite, you need to run it through a local development server. 

### Prerequisites
Make sure you have [Node.js](https://nodejs.org/) installed on your machine.


