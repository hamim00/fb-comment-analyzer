import streamlit as st
import matplotlib.pyplot as plt

def ensure_engagement_columns(df):
    reaction_cols = ['like_count', 'love_count', 'haha_count', 'wow_count', 'sad_count', 'angry_count', 'care_count']
    if 'total_reactions' not in df.columns:
        df['total_reactions'] = df[reaction_cols].sum(axis=1)
    if 'engagement_score' not in df.columns:
        if 'reply_count' in df.columns:
            df['engagement_score'] = df['total_reactions'] + df['reply_count']
        else:
            df['engagement_score'] = df['total_reactions']  # fallback

def display_kpis_and_highlights(df):
    ensure_engagement_columns(df)
    st.subheader("🔢 Key Metrics")
    total_comments = len(df)
    unique_users = df['user_id'].nunique() if 'user_id' in df else df['user_name'].nunique()
    reaction_cols = ['like_count', 'love_count', 'haha_count', 'wow_count', 'sad_count', 'angry_count', 'care_count']
    total_reactions = int(df[reaction_cols].sum(axis=1).sum())
    avg_engagement = float(df[reaction_cols].sum(axis=1).mean())
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Comments", total_comments)
    kpi2.metric("Unique Users", unique_users)
    kpi3.metric("Total Reactions", total_reactions)
    kpi4.metric("Avg Reactions/Comment", f"{avg_engagement:.1f}")

    st.subheader("🔥 Top Engagement Highlights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Top 5 Most Engaged Comments**")
        st.dataframe(df.sort_values('engagement_score', ascending=False)[['comment_text', 'engagement_score']].head(5), use_container_width=True)
    with col2:
        st.markdown("**Top 5 Most Loved Comments**")
        st.dataframe(df.sort_values('love_count', ascending=False)[['comment_text', 'love_count']].head(5), use_container_width=True)
    with col3:
        st.markdown("**Top 5 Most Haha-ed Comments**")
        st.dataframe(df.sort_values('haha_count', ascending=False)[['comment_text', 'haha_count']].head(5), use_container_width=True)

def display_dashboard(df):
    ensure_engagement_columns(df)
    st.header("📊 Facebook Comment Analytics Dashboard")
    emotion_col = 'emotion_simple' if 'emotion_simple' in df.columns else 'emotion'

    st.subheader("🔠 NLP Analysis Metrics")
    kpi1, kpi2 = st.columns(2)
    most_common_sentiment = df['sentiment'].mode().iloc[0] if 'sentiment' in df else 'N/A'
    most_common_emotion = df[emotion_col].mode().iloc[0] if emotion_col in df else 'N/A'
    kpi1.metric("Most Common Sentiment", most_common_sentiment)
    kpi2.metric("Most Common Emotion", most_common_emotion)

    st.markdown("---")

    st.subheader("📝 All Comments with Sentiment & Emotion")
    show_cols = [c for c in ['user_name', 'comment_text', 'sentiment', emotion_col, 'total_reactions', 'engagement_score'] if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True)

    st.markdown("---")

    st.subheader("📈 Engagement by Sentiment & Emotion")
    graph_col1, graph_col2 = st.columns(2)
    with graph_col1:
        st.markdown("**Engagement by Sentiment**")
        if 'sentiment' in df and 'engagement_score' in df:
            sentiment_engagement = df.groupby('sentiment')['engagement_score'].sum().sort_values(ascending=False)
            fig1, ax1 = plt.subplots()
            sentiment_engagement.plot(kind='bar', ax=ax1)
            ax1.set_ylabel("Engagement Score")
            ax1.set_xlabel("Sentiment")
            st.pyplot(fig1)
    with graph_col2:
        st.markdown("**Engagement by Emotion**")
        if emotion_col in df and 'engagement_score' in df:
            emotion_engagement = df.groupby(emotion_col)['engagement_score'].sum().sort_values(ascending=False)
            fig2, ax2 = plt.subplots()
            emotion_engagement.plot(kind='bar', ax=ax2, color='orange')
            ax2.set_ylabel("Engagement Score")
            ax2.set_xlabel("Emotion")
            st.pyplot(fig2)
