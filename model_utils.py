from transformers import pipeline

def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        framework="pt"
    )

def load_emotion_model():
    return pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        framework="pt"
    )

def extract_top_label(preds):
    # Handle if it's a list of lists (flatten one level)
    while isinstance(preds, list) and len(preds) == 1 and isinstance(preds[0], list):
        preds = preds[0]
    # Now try to extract as usual
    if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict) and 'label' in preds[0]:
        return preds[0]['label']
    elif isinstance(preds, dict) and 'label' in preds:
        return preds['label']
    else:
        return 'neutral'

def analyze_comments(df, sentiment_model, emotion_model):
    df = df.copy()
    # Sentiment
    df['sentiment'] = df['comment_text'].apply(lambda x: sentiment_model(x)[0]['label'])
    # Emotion
    df['emotion_raw'] = df['comment_text'].apply(lambda x: emotion_model(x))
    df['emotion'] = df['emotion_raw'].apply(extract_top_label)
    return df
