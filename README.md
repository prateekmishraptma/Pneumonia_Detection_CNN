## 🧠 Pneumonia Detection using CNN

This project builds a Convolutional Neural Network (CNN) to classify chest X-ray images into:

* **PNEUMONIA**
* **NORMAL**

---

## 🚀 Project Workflow

1. **Data Loading**

   * Images are loaded from directory structure
   * Labels are assigned using folder names

2. **Preprocessing**

   * Convert images to grayscale
   * Resize images to 150x150
   * Normalize pixel values (0–255 → 0–1)

3. **Model Architecture**

   * Conv2D layers (32 → 64 → 64 → 128 → 256)
   * Batch Normalization
   * MaxPooling
   * Dropout (regularization)
   * Flatten layer
   * Dense(128, ReLU)
   * Dense(1, Sigmoid)

4. **Training**

   * Optimizer: Adam
   * Loss: Binary Crossentropy
   * Metrics: Accuracy

5. **Prediction**

   * Output probabilities converted to binary using threshold 0.5

---

## 📊 Model Summary
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/4e45acb4-1a7c-4e08-ad8a-af5a59155f45" />

* Input: (150, 150, 1)
* Output: Binary classification
* Total Params: ~1.24M

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook pneumoniaDetection.ipynb
```

---

## 📌 Key Concepts Used

* Convolutional Neural Networks (CNN)
* Data Augmentation
* Batch Normalization
* Dropout (Overfitting prevention)
* Sigmoid activation (Binary classification)

---

## 🎯 Output Interpretation

```python
(model.predict(x_test) > 0.5).astype("int32")
```

* 0 → Pneumonia
* 1 → Normal

---

## 📁 requirements.txt

```
numpy
pandas
matplotlib
opencv-python
tensorflow
keras
scikit-learn
```
