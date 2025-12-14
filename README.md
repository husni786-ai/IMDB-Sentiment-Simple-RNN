# 💬 IMDB Sentiment Analysis with Simple Recurrent Neural Network (SimpleRNN)

This project contains a Jupyter Notebook that demonstrates an end-to-end Natural Language Processing (NLP) workflow for **Sentiment Analysis** using a **Simple Recurrent Neural Network (SimpleRNN)** built with TensorFlow/Keras. The model is trained on the classic **IMDb movie review dataset** to classify text as either positive or negative.

---

## 🎯 Project Goals

The objective is to showcase the complete pipeline for a fundamental sequence classification task:

1.  **Data Acquisition:** Load the IMDb dataset (pre-processed and pre-tokenized) directly from Keras.
2.  **Preprocessing:** Prepare the sequence data by padding and limiting vocabulary size.
3.  **Model Architecture:** Define a sequential model featuring the key **Embedding** and **SimpleRNN** layers.
4.  **Regularization:** Incorporate **Dropout** and use **Early Stopping** to prevent overfitting.
5.  **Training & Evaluation:** Compile, train, and evaluate the final model performance.

## ⚙️ Technology Stack and Dependencies

The project relies on standard deep learning and data processing libraries:

| Library | Role |
| :--- | :--- |
| **`tensorflow / keras`** | Core framework for deep learning model construction and training. |
| **`datasets.imdb`** | Used to load the pre-tokenized IMDb dataset. |
| **`preprocessing.sequence`** | Used for padding sequences to a uniform length (`pad_sequences`). |
| **`layers.Embedding`** | Converts integer-encoded words into dense, fixed-size vectors. |
| **`layers.SimpleRNN`** | The core recurrent layer for processing sequences.  |
| **`callbacks.EarlyStopping`** | Stops training automatically when performance plateaus on the validation set. |

### Installation

```bash
pip install tensorflow numpy
