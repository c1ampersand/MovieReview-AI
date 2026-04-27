# MovieReview AI

MovieReview AI is an AI-powered movie review sentiment analyzer. The system predicts whether a movie review is positive or negative using natural language processing and machine learning.

## Project Overview

The goal of this project is to create a real-world AI system that allows users to enter a movie review and receive a sentiment prediction. The app displays the predicted sentiment, a confidence score, and a simple explanation of the model output.

## Dataset

This project uses the IMDB movie review dataset. The dataset contains labeled movie reviews that are classified as either positive or negative.

For this class project, the model was trained on 10,000 reviews and tested on 3,000 reviews.

## AI Method

The system uses:

- TF-IDF Vectorizer for text feature extraction
- Logistic Regression for sentiment classification
- Streamlit for the interactive web application

## Features

- Movie review text input
- Positive or negative sentiment prediction
- Confidence score
- Explanation using influential words or phrases
- Model performance page
- Confusion matrix
- Responsible AI section

## Model Performance

The model reached about 89% accuracy on the test set.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the app:

```bash
streamlit run app.py
```

## Responsible AI

MovieReview AI is not perfect. It may struggle with sarcasm, slang, jokes, very short reviews, or reviews that contain mixed opinions. The app includes confidence scores and explanations to make the system more transparent.