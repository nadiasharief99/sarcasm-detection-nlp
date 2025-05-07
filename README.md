# 🧠 Sarcasm Detection from Headlines Using Machine Learning

**Author:** Nadia Sharief  
**Course:** INFO 539 – Natural Language Processing  
**Semester:** Spring 2025

---

## 📌 Project Overview

This project focuses on **detecting sarcasm in news headlines** using a variety of machine learning approaches. It progresses from traditional models like **Logistic Regression** and **Random Forest** to more advanced **LSTM** neural networks and a fine-tuned **BERT** transformer. The goal is to compare and evaluate the performance of these models using standard classification metrics.

---

## 📦 Dataset Information

- **Name:** [Sarcasm Headlines Dataset v2](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- **Size:** ~28,000 labeled headlines (`1` for sarcastic, `0` for not sarcastic)
- **Citation:**
  - Misra & Arora, *AI Open* (2023)
  - Misra & Grover, *Sculpting Data for ML* (2021)

> ⚠️ **Note:** The dataset file `Sarcasm_Headlines_Dataset_v2.json` is **not included** in the repository. Upload it to your Google Drive and update the path accordingly in the notebook.

---

## ⚙️ Environment Notes

This project was developed in Google Colab to take advantage of free GPU acceleration.
The LSTM and especially the BERT model require substantial compute resources. In particular, fine-tuning BERT took over 5 hours per epoch, making Colab an ideal    environment for experimentation and reproducibility.While the code can run on a local machine, execution time and memory requirements may be limiting without a dedicated GPU.

---

## 🔧 Setup Instructions

### 📁 Mount Google Drive (Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 📍 Define Dataset Path
```python
DATA_PATH = "/content/drive/MyDrive/Sarcasm_Headlines_Dataset_v2.json"
```

### 📦 Install Required Libraries
```bash
!pip install -q nltk transformers datasets
```

> You may see some dependency warnings — these are safe to ignore unless you're using GPU-heavy Torch operations.

---

## 🔎 Exploratory Data Analysis (EDA)

- Cleaned headlines using lowercasing, punctuation removal, stopword filtering, and stemming.
- Visualized class distribution to confirm near-balance between sarcastic and non-sarcastic labels.

---

## 🧪 Models Implemented

### 🔹 Logistic Regression (TF-IDF)
- **Accuracy:** 79%
- **F1-Score:** 0.79  
- **Strengths:** Fast, interpretable  
- **Weaknesses:** Limited contextual understanding

---

### 🔹 Random Forest (TF-IDF)
- **Accuracy:** 76%
- **F1-Score:** 0.76  
- **Strengths:** Handles non-linearity, class imbalance  
- **Weaknesses:** Weaker sarcasm detection vs Logistic Regression

---

### 🔹 LSTM Neural Network
- **Accuracy:** 81%
- **F1-Score:** 0.81  
- **Strengths:** Captures word sequences  
- **Weaknesses:** Slower, less context-aware than transformers

---

### 🔹 Fine-Tuned BERT (2 epochs)
- **Accuracy:** 83%
- **F1-Score:** 0.83  
- **Strengths:** Best overall performance; strong contextual understanding  
- **Weaknesses:** High compute time (~5 hours/epoch)

---

## 📊 Performance Comparison

| Model               | Accuracy | F1-Score |
|--------------------|----------|----------|
| Logistic Regression| 79%      | 0.79     |
| Random Forest      | 76%      | 0.76     |
| LSTM               | 81%      | 0.81     |
| BERT               | **83%**  | **0.83** |

---

## ✅ Conclusion

This project demonstrates how sarcasm detection improves as we move from traditional ML to deep learning and transformer-based architectures. **BERT** achieved the highest performance, showing its effectiveness for context-rich tasks.

The entire pipeline—from preprocessing to model training and evaluation—is designed to be **modular, reproducible**, and adaptable for future text classification challenges.
