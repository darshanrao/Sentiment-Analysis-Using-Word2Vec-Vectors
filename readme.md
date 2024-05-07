# Advanced Sentiment Analysis on Amazon Reviews

## Overview
This project is an extension of a previous assignment on sentiment analysis, now incorporating advanced neural models and word embeddings. The focus is on employing deep learning techniques to enhance the classification accuracy of sentiment analysis on Amazon kitchen product reviews. This assignment is part of the CSCI544 course at the University of Southern California.

## Dataset
The Amazon Reviews dataset from HW1 but extend it to include balanced data across all rating categories. A total of 250,000 reviews are randomly selected, ensuring an even distribution across positive, neutral, and negative sentiments.

## Feature Engineering and Models
### Word Embedding
1. **Pretrained Word2Vec**: Utilizes the `word2vec-google-news-300` model to derive word embeddings.
2. **Custom Word2Vec**: A model trained on the dataset using Gensim with specific hyperparameters (300 dimensions, window size of 11, minimum word count of 10).

### Machine Learning Models
- **Simple Models**: Perceptron and SVM are trained using average Word2Vec vectors of the reviews.
- **Feedforward Neural Networks (FNN)**: A multilayer perceptron network with two hidden layers, trained for both binary and ternary classification.
- **Convolutional Neural Networks (CNN)**: A simple CNN setup to handle sequence data from reviews, trained similarly in binary and ternary modes.

## Requirements
- Python 3
- Pandas
- Gensim
- PyTorch or TensorFlow/Keras (depending on the chosen implementation)
- Scikit-learn

## File Structure
- `amazon_reviews_us_Office_Products_v1_00.tsv.gz`: The dataset file.
- `homework.ipynb`: Jupyter Notebook containing the full implementation and analysis.
- `README.md`: This file, explaining the project setup and execution.

## Usage
1. Ensure all dependencies are installed and Python 3 is available.
2. Download the dataset and ensure it is located in the same directory as the notebook.
3. Open and run `homework.ipynb` to execute the data preparation, model training, and evaluation steps.


