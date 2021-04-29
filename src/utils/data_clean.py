import re
import nltk
import datetime

import pandas as pd

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter


def data_clean(load_path:str, save_path:str):
    # Download Necessory Component
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    stopword_list = nltk.corpus.stopwords.words('english')

    # Read Dataset
    tweets = pd.read_csv(load_path).drop(['Unnamed: 0'],axis=1)
    tweets.date = pd.to_datetime(tweets.date)

    # Clean Data
    tweets = clean(tweets)

    # Lemmatizer
    tweets.cleaned_tweet = data_lemmatizer(tweets.cleaned_tweet, lemmatizer)

    # Remove Useless Words
    tweets.cleaned_tweet = tweets.cleaned_tweet.map(lambda x: x.replace('#',''))
    tweets.cleaned_tweet = tweets.cleaned_tweet.map(lambda x: x.replace('q',''))
    tweets.cleaned_tweet = tweets.cleaned_tweet.map(lambda x: x.replace('tesla',''))
    tweets.cleaned_tweet = tweets.cleaned_tweet.map(lambda x: x.replace('tsla',''))

    # Save 
    tweets.to_csv(save_path, index=False)


def clean(dataframe):
    # Add whitespace to the end of every tweet
    dataframe['cleaned_tweet'] = dataframe.tweet.map(lambda x: x + " ") 
    
    # Remove http links
    dataframe.cleaned_tweet = dataframe.cleaned_tweet.map(lambda x: re.sub(r'http.*', '', x))
    
    # Remove special characters and numbers
    dataframe.cleaned_tweet = dataframe.cleaned_tweet.map(lambda x: re.sub(r"[^a-zA-Z#]", ' ', x))
    
    # Lowercase all tweets
    dataframe.cleaned_tweet = dataframe.cleaned_tweet.map(lambda x: x.lower())
    
    #Tokenize tweets and remove stop words
    stopword_list = stopwords.words('english')
    for i in range(len(dataframe.cleaned_tweet)):
        tokens = word_tokenize(dataframe.cleaned_tweet[i])
        clean_tokens = [w for w in tokens if w not in stopword_list]
        dataframe.cleaned_tweet[i] = clean_tokens

    return dataframe


def data_lemmatizer(tweets, lemmatizer):
    for i in range(len(tweets)):
        # Pos-tag each word in tweet
        for word in [tweets[i]]:
            pos_tag_list = nltk.pos_tag(word)
        
        # Convert pos-tag to be wordnet compliant
        wordnet_tags = []
        for j in pos_tag_list:
            # Adjective
            if j[1].startswith('J'):
                wordnet_tags.append(wordnet.ADJ)
            
            # Noun
            elif j[1].startswith('N'):
                wordnet_tags.append(wordnet.NOUN)
            
            # Adverb
            elif j[1].startswith('R'):
                wordnet_tags.append(wordnet.ADV)
            
            # Verb
            elif j[1].startswith('V'):
                wordnet_tags.append(wordnet.VERB)
            
            # Default to noun
            else:
                wordnet_tags.append(wordnet.NOUN)
        
        # Lemmatize each word in tweet
        lem_words = []
        for k in range(len(tweets[i])):
            lem_words.append(lemmatizer.lemmatize(tweets[i][k], pos=wordnet_tags[k]))
        lem_tweet = ' '.join(lem_words)
        tweets[i] = lem_tweet

    return tweets