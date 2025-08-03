import streamlit as st
from fetch_comments import fetch_comments
from model_utils import load_sentiment_model, load_emotion_model, analyze_comments
from dashboard_components import display_dashboard, display_kpis_and_highlights

st.set_page_config(page_title="FB Comment Analyzer", layout="wide", initial_sidebar_state="collapsed")

query_params = st.query_params
fb_token = query_params.get("fb_token", None)
post_url = query_params.get("post_url", None)


# ... your token/post_url fetching logic ...

if fb_token and post_url:
    # Step 1: Fetch comments (no analysis yet)
    comments = None
    with st.spinner("Fetching comments from Facebook..."):
        comments = fetch_comments(fb_token, post_url)

    if comments is not None and not comments.empty:
        st.success(f"Fetched {len(comments)} comments. Preview below:")

        # SHOW ONLY basic KPIs & highlights for now!
        display_kpis_and_highlights(comments)

        # Step 2: Now run analysis
        with st.spinner("Analyzing comments (sentiment & emotion)..."):
            sentiment_model = load_sentiment_model()
            emotion_model = load_emotion_model()
            analyzed_df = analyze_comments(comments, sentiment_model, emotion_model)

        st.success("Analysis complete! See below:")

        # Full dashboard
        display_dashboard(analyzed_df)
    else:
        st.error("No comments found or error fetching comments for this post.")
