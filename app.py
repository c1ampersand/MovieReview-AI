import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="MovieReview AI",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load("models/movie_review_model.pkl")


@st.cache_data
def load_metrics():
    return pd.read_csv("data/model_metrics.csv")


def predict_sentiment(model, review_text):
    prediction = model.predict([review_text])[0]
    probabilities = model.predict_proba([review_text])[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[prediction] * 100

    return sentiment, confidence, probabilities


def get_model_explanation(model, review_text, sentiment):
    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]

        transformed_text = vectorizer.transform([review_text])
        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]

        scores = transformed_text.toarray()[0] * coefficients

        if sentiment == "Positive":
            top_indexes = scores.argsort()[-8:][::-1]
            explanation_type = "These words or phrases pushed the model more toward a positive prediction:"
        else:
            top_indexes = scores.argsort()[:8]
            explanation_type = "These words or phrases pushed the model more toward a negative prediction:"

        important_words = []

        for index in top_indexes:
            if scores[index] != 0:
                important_words.append(feature_names[index])

        if len(important_words) == 0:
            important_words = ["No strong individual keywords found"]

        return explanation_type, important_words

    except Exception:
        return "Simple explanation:", ["The model used text patterns from the full review to make this prediction."]


if not os.path.exists("models/movie_review_model.pkl"):
    st.error("Model file not found. Run this first in the terminal: python train_model.py")
    st.stop()

if not os.path.exists("data/model_metrics.csv"):
    st.error("Metrics file not found. Run this first in the terminal: python train_model.py")
    st.stop()


model = load_model()
metrics = load_metrics()


st.title("🎬 MovieReview AI")
st.subheader("AI-Powered Movie Review Sentiment Analyzer")

st.write(
    "MovieReview AI uses natural language processing and machine learning to predict whether a movie review is positive or negative."
)

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Sentiment Predictor", "Model Performance", "Responsible AI"]
)


if menu == "Home":
    st.header("Project Overview")

    st.write(
        """
        **MovieReview AI** is a real-world AI system that analyzes movie reviews and predicts sentiment.

        The goal of this project is to allow users to type in a movie review and quickly see whether the review is likely positive or negative.
        The system also shows a confidence score and a simple explanation of the prediction.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Task", "Sentiment Classification")

    with col2:
        st.metric("Model", "Logistic Regression")

    with col3:
        st.metric("Text Method", "TF-IDF")

    st.header("How the System Works")

    st.write(
        """
        1. The user enters a movie review.
        2. The system converts the review text into numerical features using TF-IDF.
        3. A Logistic Regression model predicts whether the review is positive or negative.
        4. The app displays the prediction, confidence score, and explanation.
        """
    )

    st.header("Dataset")

    st.write(
        """
        This project uses the IMDB movie review dataset. The dataset contains labeled movie reviews that are marked as either positive or negative.
        The model was trained on 10,000 reviews and tested on 3,000 reviews for this class project.
        """
    )


elif menu == "Sentiment Predictor":
    st.header("Try the Sentiment Predictor")

    sample_review = st.selectbox(
        "Choose a sample review or type your own below:",
        [
            "",
            "The movie was amazing and emotional with excellent acting.",
            "The film was boring, slow, and badly written.",
            "I loved the characters and the story was beautiful.",
            "The plot made no sense and the acting was terrible.",
            "The acting was good, but the story was boring and too long."
        ]
    )

    review_text = st.text_area(
        "Enter a movie review:",
        value=sample_review,
        height=180,
        placeholder="Example: The movie was exciting, emotional, and had amazing acting."
    )

    if st.button("Analyze Review"):
        if review_text.strip() == "":
            st.warning("Please enter a movie review first.")
        else:
            sentiment, confidence, probabilities = predict_sentiment(model, review_text)

            st.subheader("Prediction Result")

            if sentiment == "Positive":
                st.success(f"Prediction: {sentiment}")
            else:
                st.error(f"Prediction: {sentiment}")

            st.metric("Confidence", f"{confidence:.2f}%")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Negative Probability", f"{probabilities[0] * 100:.2f}%")

            with col2:
                st.metric("Positive Probability", f"{probabilities[1] * 100:.2f}%")

            st.subheader("Model Explanation")

            explanation_type, important_words = get_model_explanation(model, review_text, sentiment)

            st.write(explanation_type)
            st.write(", ".join(important_words))

            st.caption(
                "This explanation is based on words or phrases from the review that influenced the model. The full model uses TF-IDF text patterns, not only these words."
            )


elif menu == "Model Performance":
    st.header("Model Performance")

    accuracy = metrics.loc[0, "accuracy"]

    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Classification Metrics")

    performance_table = pd.DataFrame({
        "Class": ["Negative", "Positive"],
        "Precision": [
            metrics.loc[0, "negative_precision"],
            metrics.loc[0, "positive_precision"]
        ],
        "Recall": [
            metrics.loc[0, "negative_recall"],
            metrics.loc[0, "positive_recall"]
        ],
        "F1-Score": [
            metrics.loc[0, "negative_f1"],
            metrics.loc[0, "positive_f1"]
        ]
    })

    st.dataframe(performance_table, use_container_width=True)

    st.subheader("Confusion Matrix")

    tn = int(metrics.loc[0, "true_negative"])
    fp = int(metrics.loc[0, "false_positive"])
    fn = int(metrics.loc[0, "false_negative"])
    tp = int(metrics.loc[0, "true_positive"])

    confusion_matrix_table = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )

    st.dataframe(confusion_matrix_table, use_container_width=True)

    fig, ax = plt.subplots()
    matrix_values = [[tn, fp], [fn, tp]]

    ax.imshow(matrix_values)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Negative", "Pred Positive"])
    ax.set_yticklabels(["Actual Negative", "Actual Positive"])
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix_values[i][j]), ha="center", va="center")

    st.pyplot(fig)

    st.write(
        """
        The confusion matrix shows how many reviews the model classified correctly and incorrectly.
        True negatives and true positives are correct predictions.
        False positives and false negatives are mistakes.
        """
    )


elif menu == "Responsible AI":
    st.header("Responsible and Explainable AI")

    st.subheader("Transparency")

    st.write(
        """
        MovieReview AI shows the predicted sentiment and a confidence score. This helps users understand how certain the model is instead of only seeing a positive or negative label.
        """
    )

    st.subheader("Explainability")

    st.write(
        """
        The app shows words or phrases from the review that influenced the prediction. This makes the model output easier to understand for users.
        """
    )

    st.subheader("Limitations")

    st.write(
        """
        This model is not perfect. It may struggle with sarcasm, slang, jokes, short reviews, or reviews that contain both positive and negative opinions.
        """
    )

    st.info(
        'Example: A review like "This movie was so bad it was actually good" may confuse the model because it contains both negative and positive wording.'
    )

    st.subheader("Ethical Use")

    st.write(
        """
        This system should be used as a support tool, not as a final authority. It only analyzes review text and does not use personal identity information.
        """
    )