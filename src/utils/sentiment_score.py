import re
import nltk
import pandas as pd
import seaborn as sns

from textblob import TextBlob


def sentiment_score(load_path:str, save_path:str):
    tweets = pd.read_csv(load_path)
    tweets.date = pd.to_datetime(tweets.date)
    tweets = tweets.dropna()

    # Calculate Sentiment Score
    tweets['sentiment'] = tweets.apply(lambda row: TextBlob(row.cleaned_tweet).sentiment[0], axis=1)

    # Calculate Daily Sentiment
    daily_sentiment = tweets[['date', 'sentiment']].resample('D', on='date').mean()

    # Clean 0 Sentiment Score
    clean = tweets.copy().rename(columns={'sentiment': 's_no_0'})
    clean = clean[clean.s_no_0 != 0]
    clean = clean.groupby(['date'], as_index=False).mean()

    # Save
    daily_sentiment['sentiment_final'] = clean.s_no_0.values
    daily_sentiment.to_csv(save_path)
    