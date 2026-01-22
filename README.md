# 🧠 Automated Retina Disease Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** to automatically classify retinal Optical Coherence Tomography (OCT) images into multiple disease categories. The system aims to assist in early detection of retinal diseases by leveraging deep learning–based image classification.

---

## 📌 Project Overview

Retinal diseases such as **CNV, DME, DRUSEN**, and **NORMAL** conditions can be identified through OCT scans. Manual diagnosis is time-consuming and requires expert ophthalmologists.  
This project automates the classification process using a CNN model trained on labeled retinal OCT images.

---

## 🧪 Dataset

- **Source**: Kaggle – *retinal-oct-images-small-version*
- **Classes**:
  - CNV (Choroidal Neovascularization)
  - DME (Diabetic Macular Edema)
  - DRUSEN
  - NORMAL
- **Data Type**: Grayscale retinal OCT images
- **Preprocessing**:
  - Image resizing
  - Normalization
  - Train–validation split

---

## ⚙️ Methodology

1. **Data Preprocessing**
   - Images resized to a fixed dimension
   - Pixel values normalized
   - Labels encoded

2. **Model Architecture**
   - Convolutional layers for feature extraction
   - MaxPooling layers for dimensionality reduction
   - Fully connected layers for classification
   - Softmax activation for multi-class output

3. **Training**
   - Loss function: Categorical Crossentropy
   - Optimizer: Adam
   - Evaluation metric: Accuracy

4. **Evaluation**
   - Model tested on validation data
   - Achieved ~84% classification accuracy

---

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Libraries & Frameworks**:
  - TensorFlow / Keras
  - NumPy
  - Pandas
  - Matplotlib
  - OpenCV
- **Environment**: Jupyter Notebook

---

## 📈 Results

- Successfully classified retinal OCT images into four categories
- Achieved approximately **84% accuracy**
- Demonstrates the effectiveness of CNNs in medical image analysis

---

## 🚀 Future Enhancements

- Improve accuracy using deeper CNN architectures
- Add data augmentation for better generalization
- Deploy the model as a web application for real-time predictions
- Integrate explainability techniques (Grad-CAM)

---

## 📂 Project Structure

├── dataset/
│ ├── train/
│ ├── test/
├── notebooks/
│ └── retina_classification.ipynb
├── models/
│ └── cnn_model.h5
├── README.md

---

## 👩‍💻 Author

**Mokshita**  
2nd Year B.Tech Student, VIT Chennai  
Interested in Machine Learning, Deep Learning, and Applied AI

---

## 📜 License

This project is intended for educational and research purposes.
