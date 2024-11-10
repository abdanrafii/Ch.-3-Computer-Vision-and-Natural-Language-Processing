---

# Ch.3: Computer Vision and Natural Language Processing

This repository contains team-based assignments on Computer Vision and Natural Language Processing (NLP), focusing on practical applications in image enhancement, object detection, and text classification. The tasks utilize advanced techniques such as transfer learning and pretrained models to address real-world problems.

---

## Notebooks Overview

### 1. Image Enhancement Techniques (`03_Kelompok_B_1.ipynb`)

This notebook explores image enhancement algorithms to improve photo quality, even under low-light conditions, by simulating smartphone camera technologies.

- **Goal**: Apply **Max Pooling** and **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance the visual quality of low-light images.
  
- **Key Steps**:
  1. **Basic Image Processing**: Convert images to grayscale and binary formats, visualize histograms.
  2. **Pooling Techniques**: Experiment with Max Pooling, Min Pooling, and Average Pooling.
  3. **Image Contrast Enhancement**: Use CLAHE for enhanced contrast in dark images.
  4. **Final Image Save**: Save enhanced images with consistent labeling.

---

### 2. Transfer Learning for Handwritten Digit Classification (`02_Kelompok_B_2.ipynb`)

This notebook implements transfer learning on pre-trained models to recognize handwritten digits using the MNIST dataset.

- **Goal**: Fine-tune pretrained models (**ResNet18**, **DenseNet121**, **Vision Transformer (ViT)**) to classify handwritten digits.
  
- **Key Steps**:
  1. **Model Customization**: Modify the input and output layers for MNIST.
  2. **Hyperparameter Tuning**: Optimize batch size, learning rate, and layer freezing.
  3. **Training and Evaluation**: Train the model and assess accuracy and loss on validation data.

---

### 3. Object Detection using YOLOv5 (`02_Kelompok_B_3.ipynb`)

This notebook demonstrates real-time object detection in video streams using a pretrained YOLOv5 model on videos sourced from YouTube.

- **Goal**: Detect objects in real-time by drawing bounding boxes and labels on detected objects.
  
- **Key Steps**:
  1. **Video Retrieval**: Capture video frames from YouTube URLs.
  2. **YOLOv5 Detection**: Use a pretrained YOLOv5 model to detect objects frame-by-frame.
  3. **Performance Assessment**: Evaluate detection accuracy and FPS for real-time performance.

---

### 4. Disaster Tweet Classification with BERT (`02_Kelompok_B_4.ipynb`)

This notebook classifies tweets related to disasters using BERT, aiming to support disaster response by analyzing public tweets.

- **Goal**: Fine-tune **BERT** for binary classification to detect disaster-related tweets.
  
- **Key Steps**:
  1. **Data Preprocessing**: Clean tweets by removing URLs, HTML tags, and stopwords.
  2. **Tokenization**: Convert text data to tokens for BERT processing.
  3. **Model Training and Evaluation**: Train BERT, validate accuracy, and assess model performance on disaster tweet classification.

---

## Running the Notebooks

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload required datasets and follow instructions within each notebook for additional dependencies.
3. Run cells in sequence for a structured analysis and model training process.

---
