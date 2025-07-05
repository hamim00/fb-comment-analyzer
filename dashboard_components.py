import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def display_dashboard(df):
    st.header("📊 Facebook Comment Analytics Dashboard")

    # Calculate additional columns
    reaction_cols = ['like_count', 'love_count', 'haha_count', 'wow_count', 'sad_count', 'angry_count', 'care_count']
    df['total_reactions'] = df[reaction_cols].sum(axis=1)
    df['engagement_score'] = df['total_reactions'] + df['reply_count']
    emotion_col = 'emotion_simple' if 'emotion_simple' in df.columns else 'emotion'

    ### 1. KPIs
    total_comments = len(df)
    unique_users = df['user_id'].nunique()
    total_reactions = int(df['total_reactions'].sum())
    avg_engagement = float(df['engagement_score'].mean())
    most_common_sentiment = df['sentiment'].mode().iloc[0] if 'sentiment' in df else 'N/A'
    most_common_emotion = df[emotion_col].mode().iloc[0] if emotion_col in df else 'N/A'

    st.subheader("🔢 Key Metrics")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Total Comments", total_comments)
    kpi2.metric("Unique Users", unique_users)
    kpi3.metric("Total Reactions", total_reactions)
    kpi4.metric("Avg Engagement", f"{avg_engagement:.1f}")
    kpi5.metric("Top Sentiment", most_common_sentiment)
    kpi6.metric("Top Emotion", most_common_emotion)

    ### 2. Top Engagements
    st.subheader("🔥 Top Engagement Highlights")
    col1, col2, col3 = st.columns(3)

    # Top 5 engaged comments
    with col1:
        st.markdown("**Top 5 Most Engaged Comments**")
        st.dataframe(df.sort_values('engagement_score', ascending=False)[['comment_text', 'engagement_score', 'sentiment', emotion_col]].head(5), use_container_width=True)
    # Top 5 loved
    with col2:
        st.markdown("**Top 5 Most Loved Comments**")
        st.dataframe(df.sort_values('love_count', ascending=False)[['comment_text', 'love_count', 'sentiment', emotion_col]].head(5), use_container_width=True)
    # Top 5 haha
    with col3:
        st.markdown("**Top 5 Most Haha-ed Comments**")
        st.dataframe(df.sort_values('haha_count', ascending=False)[['comment_text', 'haha_count', 'sentiment', emotion_col]].head(5), use_container_width=True)

    st.markdown("---")

    ### 3. Main Comment Table
    st.subheader("📝 All Comments with Sentiment & Emotion")
    st.dataframe(df[['user_name', 'comment_text', 'sentiment', emotion_col, 'total_reactions', 'engagement_score']], use_container_width=True)

    st.markdown("---")

    ### 4. Graphs
    st.subheader("📈 Engagement by Sentiment & Emotion")

    graph_col1, graph_col2 = st.columns(2)

    with graph_col1:
        st.markdown("**Engagement by Sentiment**")
        sentiment_engagement = df.groupby('sentiment')['engagement_score'].sum().sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        sentiment_engagement.plot(kind='bar', ax=ax1)
        ax1.set_ylabel("Engagement Score")
        ax1.set_xlabel("Sentiment")
        st.pyplot(fig1)

    with graph_col2:
        st.markdown("**Engagement by Emotion**")
        emotion_engagement = df.groupby(emotion_col)['engagement_score'].sum().sort_values(ascending=False)
        fig2, ax2 = plt.subplots()
        emotion_engagement.plot(kind='bar', ax=ax2, color='orange')
        ax2.set_ylabel("Engagement Score")
        ax2.set_xlabel("Emotion")
        st.pyplot(fig2)

    st.markdown("---")
