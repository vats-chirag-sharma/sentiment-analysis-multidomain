
import streamlit as st
import joblib
import os


# =====================================================
# LOAD MODEL
# =====================================================

current_folder = os.path.dirname(__file__)

model_path = os.path.join(
    current_folder,
    "sentiment_model.pkl"
)

tfidf_path = os.path.join(
    current_folder,
    "tfidf_vectorizer.pkl"
)

model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)


# =====================================================
# CUSTOM DESIGN / BACKGROUND
# =====================================================

st.markdown(
    """
    <style>

    /* Whole page background */
    .stApp {
        background: linear-gradient(
            135deg,
            #eef4ff 0%,
            #f7f9fc 50%,
            #edf3f8 100%
        );
    }

    /* Main content width and spacing */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 12px;
        min-height: 46px;
        font-weight: 600;
    }

    /* Review textbox */
    textarea {
        border-radius: 14px !important;
        background-color: white !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
    }

    /* Expander cards */
    [data-testid="stExpander"] {
        background-color: white;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
    }

    /* Alert / result boxes */
    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background-color: #dbe3ec;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "review_input" not in st.session_state:
    st.session_state.review_input = ""


# =====================================================
# FUNCTIONS
# =====================================================

def clear_review():
    st.session_state.review_input = ""


def set_example(text):
    st.session_state.review_input = text


# =====================================================
# TOP HEADER CARD
# =====================================================

# =====================================================
# TOP HEADER CARD
# =====================================================

st.markdown("""
<div style="
background: rgba(255,255,255,0.92);
padding: 30px 20px;
border-radius: 20px;
text-align: center;
box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
margin-bottom: 25px;
border: 1px solid #e2e8f0;
">

<div style="
font-size: 42px;
margin-bottom: 5px;
">
💬
</div>

<h1 style="
margin: 0;
color: #1e293b;
font-size: 36px;
">
AI Sentiment Analyzer
</h1>

<p style="
color: #64748b;
margin-top: 10px;
margin-bottom: 5px;
font-size: 16px;
">
Machine Learning Powered Review Classification
</p>

<p style="
color: #94a3b8;
margin: 0;
font-size: 14px;
">
TF-IDF • Logistic Regression • Natural Language Processing
</p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# INTRODUCTION
# =====================================================

st.write(
    "Enter a review below and the trained machine learning model "
    "will classify it as positive or negative."
)


# =====================================================
# EXAMPLE REVIEWS
# =====================================================

st.subheader("Try an Example")

col1, col2 = st.columns(2)

with col1:

    st.button(
        "😊 Positive Example",
        on_click=set_example,
        args=(
            "This movie was amazing and I really enjoyed watching it.",
        ),
        use_container_width=True
    )

with col2:

    st.button(
        "😞 Negative Example",
        on_click=set_example,
        args=(
            "This movie was terrible and a complete waste of time.",
        ),
        use_container_width=True
    )


# =====================================================
# REVIEW INPUT
# =====================================================

st.subheader("Analyze a Review")

review = st.text_area(
    "Enter your review:",
    key="review_input",
    placeholder="Type your review here...",
    height=160
)


# =====================================================
# BUTTONS
# =====================================================

button_col1, button_col2 = st.columns(2)

with button_col1:

    analyze = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True
    )

with button_col2:

    st.button(
        "Clear",
        on_click=clear_review,
        use_container_width=True
    )


# =====================================================
# PREDICTION
# =====================================================

if analyze:

    if review.strip() == "":

        st.warning(
            "Please enter a review before analyzing."
        )

    else:

        # Convert text using TF-IDF
        review_tfidf = tfidf.transform(
            [review]
        )

        # Predict sentiment
        prediction = model.predict(
            review_tfidf
        )[0]

        # Prediction probabilities
        probabilities = model.predict_proba(
            review_tfidf
        )[0]

        confidence = max(probabilities) * 100

        # Word count
        word_count = len(
            review.split()
        )


        st.divider()

        st.subheader("Analysis Result")


        if prediction == "positive":

            st.success(
                "😊 Positive Sentiment"
            )

        else:

            st.error(
                "😞 Negative Sentiment"
            )


        st.write(
            f"Confidence: **{confidence:.2f}%**"
        )

        st.progress(
            int(confidence)
        )


        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Prediction",
            prediction.title()
        )

        result_col2.metric(
            "Word Count",
            word_count
        )


        if confidence < 70:

            st.warning(
                "⚠️ Low confidence prediction — "
                "the model is uncertain about this review."
            )


        # Save in session history
        st.session_state.history.append(
            {
                "review": review,
                "sentiment": prediction,
                "confidence": confidence
            }
        )


# =====================================================
# SESSION HISTORY
# =====================================================

if len(st.session_state.history) > 0:

    st.divider()

    st.subheader("📊 Session Summary")

    positive_count = sum(
        1
        for item in st.session_state.history
        if item["sentiment"] == "positive"
    )

    negative_count = sum(
        1
        for item in st.session_state.history
        if item["sentiment"] == "negative"
    )


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Reviews Analyzed",
        len(st.session_state.history)
    )

    col2.metric(
        "Positive",
        positive_count
    )

    col3.metric(
        "Negative",
        negative_count
    )


    with st.expander(
        "View Recent Predictions"
    ):

        for item in reversed(
            st.session_state.history[-5:]
        ):

            st.write(
                f"**Review:** {item['review']}"
            )

            st.write(
                f"**Prediction:** "
                f"{item['sentiment'].title()}"
            )

            st.write(
                f"**Confidence:** "
                f"{item['confidence']:.2f}%"
            )

            st.divider()


# =====================================================
# MODEL INFORMATION
# =====================================================

st.divider()

with st.expander(
    "🤖 About the Model"
):

    st.write(
        "**Machine Learning Model:** Logistic Regression"
    )

    st.write(
        "**Feature Extraction:** TF-IDF"
    )

    st.write(
        "**Training Dataset:** IMDb + Amazon + Restaurant + Sentiment140"
    )

    st.write(
        "**Dataset Size:** Approximately 150,474 unique texts"
    )

    st.write(
        "**Model Accuracy:** Approximately 84.4%"
    )

    st.write(
        "**Classes:** Positive and Negative"
    )

    st.info(
        "The model was trained primarily on movie reviews. "
        "Therefore, predictions for unrelated conversational "
        "sentences may sometimes be less accurate."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <div style="
        text-align: center;
        color: #94a3b8;
        padding-top: 30px;
        padding-bottom: 10px;
        font-size: 13px;
    ">
        AI Applications Project • Sentiment Analysis using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
