import streamlit as st
from fetch_comments import fetch_comments
from model_utils import load_sentiment_model, load_emotion_model, analyze_comments
from dashboard_components import display_dashboard

st.set_page_config(page_title="FB Comment Analyzer", layout="wide")
st.title("Facebook Comment Analyzer Dashboard")

fb_token = st.sidebar.text_input("Facebook Graph API Token", type="password")
post_id = st.sidebar.text_input("Facebook Post ID")

if st.sidebar.button("Analyze Comments"):
    comments = fetch_comments(fb_token, post_id)
    if not comments.empty:
        # Use the new HuggingFace loader, no .pkl needed!
        sentiment_model = load_sentiment_model()
        emotion_model = load_emotion_model()
        analyzed_df = analyze_comments(comments, sentiment_model, emotion_model)
        display_dashboard(analyzed_df)
    else:
        st.error("No comments found or error fetching comments.")
