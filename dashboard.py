# dashboard.py
import streamlit as st
import pandas as pd

from fetch_comments import fetch_comments
from model_utils import (
    load_sentiment_model,
    load_emotion_model,
    analyze_comments,
)
from dashboard_components import (
    display_dashboard,
    display_kpis_and_highlights,
)

st.set_page_config(
    page_title="FB Comment Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Facebook Comment Analyzer")

# ---------- Query params (works for both new/old Streamlit) ----------
qp = getattr(st, "query_params", None)
if qp is None:
    qp = st.experimental_get_query_params()

def first(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return v[0]
    return v

fb_token_qp = first(qp.get("fb_token"))
post_url_qp  = first(qp.get("post_url"))

# ---------- Inputs ----------
fb_token = st.text_input(
    "Facebook access token",
    value=fb_token_qp or "",
    type="password",
    help="Use a Page access token for your own Page posts/reels (pages_read_engagement).",
)
post_url = st.text_input(
    "Facebook post URL",
    value=post_url_qp or "",
    help="Paste a full URL (supports posts, photos, videos, reels, and group posts).",
)

# Optional switches (you can wire these later if you want replies, etc.)
col_sw1, col_sw2 = st.columns([1, 1])
with col_sw1:
    show_preview = st.checkbox("Show raw comments preview", value=True)
with col_sw2:
    run_analysis = st.checkbox("Run sentiment & emotion analysis", value=True)

# ---------- Cache heavy resources ----------
@st.cache_resource(show_spinner=False)
def _get_sentiment_model():
    return load_sentiment_model()

@st.cache_resource(show_spinner=False)
def _get_emotion_model():
    return load_emotion_model()

# ---------- Action ----------
if st.button("Analyze this post"):
    if not fb_token or not post_url:
        st.warning("Please provide BOTH an access token and a post URL.")
        st.stop()

    try:
        # Step 1: Fetch comments
        with st.spinner("Fetching comments from Facebook..."):
            comments = fetch_comments(fb_token, post_url)

        if comments is None or getattr(comments, "empty", True):
            st.error("No comments found. Check token, permissions, or post visibility.")
            st.stop()

        # Normalize/ensure expected columns exist (lightweight guard)
        expected_cols = {
            "comment_id","user_id","user_name","comment_text","created_time",
            "like_count","love_count","haha_count","wow_count","sad_count","angry_count","care_count",
            "reply_count","user_profile_link","user_gender","is_verified","language","parent_comment_id"
        }
        missing = [c for c in expected_cols if c not in comments.columns]
        if missing:
            # Create any missing columns to avoid downstream crashes
            for c in missing:
                comments[c] = "" if c not in {
                    "like_count","love_count","haha_count","wow_count","sad_count","angry_count","care_count","reply_count"
                } else 0

        st.success(f"Fetched {len(comments)} comments. Preview below:")

        # Optional raw preview
        if show_preview:
            with st.expander("Raw comments (first 100 rows)"):
                st.dataframe(comments.head(100), use_container_width=True)

        # SHOW ONLY basic KPIs & highlights first
        display_kpis_and_highlights(comments)

        # Step 2: Run analysis (if enabled)
        if run_analysis:
            with st.spinner("Analyzing comments (sentiment & emotion)..."):
                sentiment_model = _get_sentiment_model()
                emotion_model   = _get_emotion_model()
                analyzed_df = analyze_comments(comments, sentiment_model, emotion_model)

            st.success("Analysis complete! See below:")
            display_dashboard(analyzed_df)
        else:
            st.info("Analysis skipped. Enable the checkbox above to run sentiment & emotion analysis.")

    except Exception as e:
        st.error(f"Fetching or processing failed: {e}")
        st.stop()
