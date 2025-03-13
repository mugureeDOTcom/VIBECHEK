import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')


# 🎯 Download VADER lexicon (only needed once)
nltk.download("vader_lexicon")

# 🎯 Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# 🎯 Step 1: App Title
st.title("📊 Customer Review Sentiment Analyzer")

# 🎯 Step 2: File Upload
uploaded_file = st.file_uploader("📂 Upload a CSV file containing reviews", type=["csv"])

if uploaded_file:
    # 🎯 Step 3: Read CSV
    df = pd.read_csv(uploaded_file)

    # 🎯 Step 4: Display Raw Data
    st.subheader("📋 Uploaded Data")
    st.write(df.head())

    # 🎯 Step 5: Define Sentiment Analysis Function
    def get_sentiment(text):
        score = sia.polarity_scores(str(text))["compound"]
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    # 🎯 Step 6: Apply Sentiment Analysis
    df["Sentiment"] = df["Cleaned_Review"].apply(get_sentiment)

    # 🎯 Step 7: Display Processed Data
    st.subheader("📊 Processed Reviews with Sentiment")
    st.write(df.head())

    # 🎯 Step 8: Sentiment Distribution
    st.subheader("📊 Sentiment Breakdown")

    sentiment_counts = df["Sentiment"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        colors=["green", "red", "blue"],
    )
    st.pyplot(fig)

    # 🎯 Step 10: AI-Based Business Recommendations
    st.subheader("📢 AI-Generated Business Recommendations")

    if "Negative" in df["Sentiment"].values:
        st.warning("🚨 Negative feedback detected! Consider improving service quality.")
    if "Positive" in df["Sentiment"].values:
        st.success("🎉 Many happy customers! Maintain excellent service.")
    if "Neutral" in df["Sentiment"].values:
        st.info("🤔 Many neutral reviews. Try engaging customers for better feedback.")

    # 🎯 Step 11: Download Processed Data
    st.subheader("📥 Download Processed Data")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="📥 Download CSV", data=csv, file_name="processed_reviews.csv", mime="text/csv")
