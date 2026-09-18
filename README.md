
# AI Sentiment Analyzer

A machine-learning web application that classifies text reviews as Positive or Negative.

## Project Workflow

Dataset → Data Cleaning → Train/Test Split → TF-IDF → Logistic Regression → Prediction → Streamlit Frontend

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

## Dataset

IMDb Dataset of 50,000 Movie Reviews.

After cleaning and removing duplicates, approximately 49,581 reviews were used.

## Model Performance

- Model: Logistic Regression
- Feature Extraction: TF-IDF
- Accuracy: approximately 89.05%
- Classes: Positive and Negative

## Web Application Features

- Review sentiment prediction
- Confidence percentage
- Low-confidence warning
- Positive and negative example reviews
- Word count
- Session prediction history
- Model information section

## Limitation

The model was trained mainly on movie reviews, so predictions on unrelated conversational text may be less accurate.
