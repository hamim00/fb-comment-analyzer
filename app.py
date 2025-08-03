import streamlit as st
from fetch_comments import fetch_comments
from model_utils import load_sentiment_model, load_emotion_model, analyze_comments
from dashboard_components import display_dashboard

# --- Sidebar Navigation ---
st.set_page_config(page_title="FB Comment Analyzer", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard"])

# --- Shared Session State for Token and Post URL ---
if "fb_token" not in st.session_state:
    st.session_state["fb_token"] = ""
if "post_url" not in st.session_state:
    st.session_state["post_url"] = ""
if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False

# --- Home Page (Login/Welcome) ---
if page == "Home":
    st.markdown("""
        <div style="text-align:center; margin-top: 2em;">
            <h1 style="font-size:2.7em; color:#1e96fc; font-weight:900; letter-spacing:-1px; margin-bottom:0.25em;">
                Facebook Comment Analyzer
            </h1>
            <p style="font-size:1.25em; color:#9cb4c9; margin-bottom:2em;">
                Welcome! Analyze your Facebook posts' comments using advanced NLP tools.<br>
                <b>Start by logging in with Facebook below.</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Facebook token input
    fb_token = st.text_input(
        "Facebook Graph API Token",
        type="password",
        value=st.session_state["fb_token"],
        key="fb_token_input"
    )
    if fb_token:
        st.session_state["fb_token"] = fb_token

    # Post URL input (optional)
    post_url = st.text_input(
        "Facebook Post URL (optional, can also fill on Dashboard)",
        value=st.session_state["post_url"],
        key="post_url_input"
    )
    if post_url:
        st.session_state["post_url"] = post_url

    st.markdown(
        """
        <div style="padding: 1em; background: #22293b; border-radius: 8px; margin-bottom: 1.5em;">
            <span style="font-size: 1.1em; color: #fff;">
                <b>Don't have a Facebook Graph API Token?</b>
                <a href="http://localhost:5000" target="_blank" style="color:#53a7ea; text-decoration: underline; margin-left:8px;">
                    Click here to log in and get your token!
                </a>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
        <hr>
        <div style="text-align:center; color:#506073; font-size:1em; margin-top:1.5em;">
            &copy; 2025 FB Comment Analyzer | Powered by <b>Streamlit</b> + <b>Flask</b>
        </div>
    """, unsafe_allow_html=True)

    st.info("Go to the **Dashboard** from the sidebar after logging in to start analyzing!")

# --- Dashboard Page ---
elif page == "Dashboard":
    st.title("Facebook Comment Analyzer Dashboard")

    # Inputs with autofill from session state
    fb_token = st.text_input(
        "Facebook Graph API Token",
        type="password",
        value=st.session_state.get("fb_token", ""),
        key="dashboard_fb_token"
    )
    post_url = st.text_input(
        "Facebook Post URL",
        value=st.session_state.get("post_url", ""),
        key="dashboard_post_url"
    )

    # Save input changes to session state
    st.session_state["fb_token"] = fb_token
    st.session_state["post_url"] = post_url

    # Trigger analysis
    if st.button("Analyze Comments"):
        if not fb_token or not post_url:
            st.error("Please enter BOTH Access Token and Post URL.")
            st.session_state["analyzed"] = False
        else:
            comments = fetch_comments(fb_token, post_url)
            if comments is not None and not comments.empty:
                sentiment_model = load_sentiment_model()
                emotion_model = load_emotion_model()
                analyzed_df = analyze_comments(comments, sentiment_model, emotion_model)
                display_dashboard(analyzed_df)
                st.session_state["analyzed"] = True
            else:
                st.error("No comments found or error fetching comments.")
                st.session_state["analyzed"] = False

    # Optionally, auto-clear analyzed state if user edits fields
    if st.session_state.get("analyzed") and (
        (fb_token != st.session_state.get("fb_token", "")) or
        (post_url != st.session_state.get("post_url", ""))
    ):
        st.session_state["analyzed"] = False
