# 😊 Facial Expression Recognition using CNNs

This project implements a Convolutional Neural Network (CNN) to classify facial expressions into four categories: **Happy**, **Sad**, **Neutral**, and **Angry**. The model is trained on a dataset of over 26,000 images and uses various deep learning techniques to improve accuracy and generalization.

---

## 📌 Project Highlights

- ✅ Built a deep CNN model for multi-class facial emotion classification
- 📈 Applied real-time data augmentation (rotation, shear, zoom, horizontal flip)
- 🧠 Incorporated dropout and batch normalization for regularization
- 📊 Evaluated model performance using accuracy, confusion matrix, and loss curves

---

## 🗂️ Dataset Overview

The dataset contains a total of **26,217 facial images** categorized into 4 emotion classes:

| Emotion  | Number of Images |
|----------|------------------|
| Happy    | 8,989            |
| Sad      | 6,077            |
| Neutral  | 6,198            |
| Angry    | 4,953            |

Data was preprocessed and augmented to improve model robustness and reduce overfitting.

---

## 🧠 Model Architecture

The CNN model consists of:

- Convolutional layers with ReLU activation
- MaxPooling layers to reduce spatial dimensions
- BatchNormalization to stabilize learning
- Dropout layers for regularization
- Fully connected Dense layers for final classification

---

## 🛠️ Tools & Technologies

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV (for image preprocessing)

---

## 📈 Results & Evaluation

- Achieved high classification accuracy on the validation set
- Visualized training/validation accuracy and loss
- Plotted confusion matrix to observe class-wise performance

---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/facial-expression-recognition-cnn.git
   cd facial-expression-recognition-cnn
