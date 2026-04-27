# MovieReview AI

MovieReview AI is an AI-powered movie review sentiment analyzer that predicts whether a written movie review is **positive** or **negative**. The system uses natural language processing and machine learning to classify review text, display a confidence score, and provide a simple explanation of the prediction.

## Contact

**Developer:** Aiden Sherlock  
**Course:** CAP 4630 - Introduction to Artificial Intelligence  
**Institution:** Florida Atlantic University  
**Project Link:** https://github.com/c1ampersand/MovieReview-AI

---

## Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [AI Methodology](#ai-methodology)
- [System Pipeline](#system-pipeline)
- [Implementation Results](#implementation-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Responsible AI](#responsible-ai)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Overview

Movie platforms and review websites contain thousands of user reviews, making it difficult to quickly understand the overall opinion of viewers. MovieReview AI helps solve this problem by automatically analyzing review text and predicting whether the review expresses a positive or negative opinion.

The project includes a machine learning model trained on movie review data and an interactive Streamlit web app where users can enter their own reviews and view the model's prediction.

---

## Project Objectives

The main goal of this project is to design and implement a complete AI system that can:

- Classify movie reviews as positive or negative
- Use natural language processing to process written text
- Display a confidence score for each prediction
- Provide a simple explanation using influential words or phrases
- Show model performance metrics such as accuracy, precision, recall, F1-score, and a confusion matrix
- Include responsible AI considerations such as transparency, limitations, and ethical use

---

## Dataset

This project uses the **Stanford Large Movie Review Dataset**, also known as the IMDB movie review dataset.

Dataset source:  
https://ai.stanford.edu/~amaas/data/sentiment/

The dataset contains labeled movie reviews for binary sentiment classification:

| Class | Meaning |
|---|---|
| Negative | The review expresses a mostly negative opinion |
| Positive | The review expresses a mostly positive opinion |

For this class project, the model was trained and tested using a smaller subset of the full dataset to keep training fast and manageable.

| Split | Samples Used |
|---|---:|
| Training Set | 10,000 reviews |
| Testing Set | 3,000 reviews |

The dataset is automatically downloaded by `train_model.py` when the model is trained.

---

## AI Methodology

MovieReview AI uses a traditional machine learning approach for natural language processing.

### Text Processing

The review text is converted into numerical features using **TF-IDF Vectorization**.

TF-IDF stands for **Term Frequency-Inverse Document Frequency**. It gives higher importance to words or phrases that are meaningful in a review while reducing the impact of common words.

### Machine Learning Model

The project uses **Logistic Regression** for binary classification.

The model predicts one of two classes:

- `0` = Negative
- `1` = Positive

### Why This Model Was Chosen

Logistic Regression with TF-IDF was chosen because it is:

- Simple to implement
- Fast to train
- Easy to explain
- Effective for text classification
- Appropriate for a student-level AI final project

---

## System Pipeline

The project follows this AI pipeline:

```text
Movie Review Text
        ↓
Text Cleaning / Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression Model
        ↓
Positive or Negative Prediction
        ↓
Confidence Score + Explanation
```

The Streamlit app allows users to interact with the trained model through a simple web interface.

---

## Implementation Results

The model was trained on 10,000 movie reviews and tested on 3,000 reviews.

### Overall Performance

| Metric | Value |
|---|---:|
| Accuracy | 89.20% |
| Negative Precision | 0.89 |
| Negative Recall | 0.90 |
| Negative F1-Score | 0.89 |
| Positive Precision | 0.90 |
| Positive Recall | 0.89 |
| Positive F1-Score | 0.89 |

### Confusion Matrix

|  | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 1348 | 152 |
| Actual Positive | 172 | 1328 |

### Interpretation

The model performs well overall, correctly classifying most positive and negative movie reviews. The accuracy of about 89% shows that the model learned useful text patterns from the dataset.

The model still makes some mistakes, especially when reviews contain mixed opinions, sarcasm, or unclear wording.

---

## Key Features

- Interactive Streamlit web app
- User text input for custom movie reviews
- Positive or negative sentiment prediction
- Confidence score for each prediction
- Negative and positive probability display
- Simple model explanation using influential words or phrases
- Model performance page
- Confusion matrix visualization
- Responsible AI section explaining limitations and ethical use

---

## Project Structure

```text
MovieReview-AI/
│
├── app.py                         # Streamlit web app
├── train_model.py                 # Model training script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Files ignored by Git
│
├── data/
│   ├── model_metrics.csv          # Saved model evaluation metrics
│   └── sample_reviews.csv         # Example reviews for testing
│
├── models/
│   └── movie_review_model.pkl     # Trained sentiment analysis model
│
└── notebooks/                     # Optional notebook folder
```

---

## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.8 or higher
- Git
- pip

### Clone the Repository

```bash
git clone https://github.com/c1ampersand/MovieReview-AI.git
cd MovieReview-AI
```

### Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the Model

Run:

```bash
python train_model.py
```

This will:

- Download the IMDB movie review dataset
- Load positive and negative reviews
- Train the TF-IDF + Logistic Regression model
- Evaluate the model
- Save the trained model to the `models/` folder
- Save performance metrics to the `data/` folder

### Run the Web App

Run:

```bash
streamlit run app.py
```

Then open the local Streamlit link in your browser.

Example:

```text
http://localhost:8501
```

---

## Example Predictions

### Positive Review Example

Input:

```text
The movie was exciting, emotional, and the acting was amazing.
```

Expected output:

```text
Prediction: Positive
Confidence: High
```

### Negative Review Example

Input:

```text
The movie was boring, slow, and the story made no sense.
```

Expected output:

```text
Prediction: Negative
Confidence: High
```

### Mixed Review Example

Input:

```text
The acting was good, but the story was boring and too long.
```

Possible output:

```text
Prediction: Positive or Negative
Confidence: Lower
```

Mixed reviews are more challenging because they contain both positive and negative language.

---

## Technologies Used

| Category | Tools |
|---|---|
| Programming Language | Python |
| Machine Learning | scikit-learn |
| NLP Feature Extraction | TF-IDF Vectorizer |
| Model | Logistic Regression |
| Data Handling | pandas, NumPy |
| Model Saving | joblib |
| Web App | Streamlit |
| Visualization | matplotlib |
| Version Control | Git and GitHub |

---

## Responsible AI

MovieReview AI includes responsible AI features to make the system more transparent and understandable.

### Transparency

The app does not only show a positive or negative label. It also displays a confidence score so users can understand how certain the model is.

### Explainability

The app shows words or phrases that influenced the model's prediction. This helps users better understand why the model made a certain decision.

### Data Privacy

The system analyzes review text only. It does not require personal identity information, user accounts, or private user data.

### Ethical Use

MovieReview AI should be used as a support tool, not as a final authority. The model can help summarize sentiment, but it should not replace human judgment.

---

## Limitations

This model is not perfect. It may struggle with:

- Sarcasm
- Slang
- Jokes
- Very short reviews
- Mixed reviews with both positive and negative opinions
- Reviews using unusual wording
- Reviews where the sentiment depends on context

Example:

```text
This movie was so bad it was actually good.
```

This type of sentence may confuse the model because it contains both negative and positive meaning.

---

## Future Improvements

Possible future improvements include:

- Training on the full IMDB dataset instead of a subset
- Adding neutral sentiment as a third class
- Using a deep learning model such as LSTM, BERT, or DistilBERT
- Adding more advanced explainability tools
- Deploying the Streamlit app online
- Adding charts showing common positive and negative words
- Allowing users to upload a CSV file of reviews for batch prediction

---

## Conclusion

MovieReview AI successfully demonstrates a complete real-world AI system for sentiment analysis. The project includes a trained machine learning model, an interactive web app, model evaluation, prediction confidence, explanation features, and responsible AI considerations.

The final model reached about **89.20% accuracy** on the test set, showing that TF-IDF and Logistic Regression can perform well for movie review sentiment classification.

---