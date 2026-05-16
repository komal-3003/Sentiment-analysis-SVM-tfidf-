# Emotion Detection using NLP and Machine Learning
## Project Objective
The objective of this project is to build an intelligent Emotion Detection system that predicts human emotions from textual data using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to analyze user-entered text and classify emotions such as joy, sadness, anger, fear, love, and surprise accurately.

## Project Description
- This project focuses on detecting emotions from text using NLP preprocessing, TF-IDF Vectorization, and a LinearSVC Machine Learning model.
- It involves text cleaning, stopword removal, tokenization, feature extraction, model training, and deployment using Streamlit.
- The system converts raw textual input into numerical vectors and predicts the corresponding emotion using a trained classification model.
- A modern interactive Streamlit UI was developed to allow real-time emotion prediction.

## Key Performance Indicators (KPIs)
- Text Input
- Emotion Classification
- NLP Preprocessing
- TF-IDF Feature Extraction
- Machine Learning Prediction
- Model Accuracy
- Real-Time Emotion Detection
## Dataset used
- <a href="https://github.com/komal-3003/Sentiment-analysis-SVM-tfidf-/blob/main/train.txt">Dataset</a>
## Project Process
## 1️.Data Collection
- The dataset contains text samples labeled with corresponding emotions.
- The dataset was used for supervised machine learning classification.
  
## 2.Data Cleaning & Preparation
- Converted text to lowercase
- Removed punctuation marks
- Removed numbers and emojis
- Tokenized text into words
- Removed stopwords while preserving important negation words like:
    - not
    - no
    - never
- Applied text normalization techniques
## 3.Exploratory Data Analysis
- Checked for null and duplicate values
- Analyzed unique emotion categories
- Explored text distribution and emotion frequency
- Studied preprocessing impact on model performance
  
## 4️.Feature Engineering
- Applied TF-IDF Vectorization for numerical text representation
-Used:
      Unigrams
      Bigrams
      Trigrams
- Extracted meaningful textual patterns for classification

## 5.Model Development
-Implemented multiple Machine Learning models:
    - Naive Bayes
    - Logistic Regression
    - LinearSVC (Support Vector Machine)
- Compared model performances using accuracy scores
- Selected LinearSVC as the final model due to superior performance

## 6.Model Evaluation
- Evaluated models using:
    - Accuracy Score
    - Classification Metrics
- Achieved approximately 89%+ prediction accuracy using LinearSVC with TF-IDF features
- Improved prediction quality using optimized preprocessing and n-gram features

## 6.Model Deployment
- Saved trained model and vectorizer using Pickle files
- Developed an interactive Streamlit web application
- Enabled users to input custom text and receive real-time emotion predictions
- Designed an attractive UI with custom CSS styling and emotion-based visual feedback

## Technologies Used
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Streamlit
- Pickle
## Emotion Detection App Interface
- <a href="https://github.com/komal-3003/Sentiment-analysis-SVM-tfidf-/blob/main/train.txt">View App</a>

## Project Insights
- TF-IDF with LinearSVC performs effectively for text classification tasks
- Proper text preprocessing significantly improves model accuracy
- Negation handling is important in sentiment and emotion analysis
- Machine Learning models can successfully classify emotions from textual patterns

## Conclusion
This project demonstrates how Natural Language Processing and Machine Learning can be combined to build an effective real-time Emotion Detection system. The model successfully predicts emotions from text with strong accuracy and provides an interactive user experience through Streamlit deployment.

## Future Scope
- Integrate Deep Learning models like LSTM and BERT
- Add speech-to-text emotion detection
- Deploy application on cloud platforms
- Improve contextual understanding using transformer models
- Add multilingual emotion detection support
