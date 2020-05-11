#!/usr/bin/env python
# coding: utf-8

# In[11]:


import flair
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
nltk.download("stopwords")
nltk.download('punkt')

a = pd.read_excel('chat.xlsx')
stop_words = set(stopwords.words('english'))
a.head()
# a.info()

b = a[['Interaction ID', 'Chat Log']]


# print(b.loc[0, 'Chat Log'])


def pre_processing(sentence):
    word_tokens = word_tokenize(sentence)
    word_tokens = [word for word in word_tokens if word.isalpha()]
    filtered_sentence = [
        w for w in word_tokens if not w.isalpha() in stop_words]
    filtered_str = ' '.join(filtered_sentence)
    return filtered_str


def predict(sentence):
    s = flair.data.Sentence(sentence)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    val = total_sentiment[0].value
    score = total_sentiment[0].score
    return val, score


l = []
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
agent_sentiment_list = []
chat_sentiment_list = []
agent_sentiment_score_list = []
chat_sentiment_score_list = []
for i, chat_log in enumerate(b['Chat Log']):
    sentence = chat_log
    conversation = sentence.split('\n')
    agent_list = []
    for sen in conversation:
        if 'Visitor' in sen and 'Client' not in sen:
            # print('visitor')
            continue
        elif 'Visitor' in sen and 'Client' in sen:
            # print("all")
            continue
        else:
            # print('client')
            temp = sen
        if temp is not None:
            agent = re.split(r'[(]+[\d]+[:]+[\d]+[:]+[\d]+[)]+[:]', temp)
            try:
                agent_list.append(agent[1])
            except:
                # print(temp)
                print(i)
    agent_str = ' '.join(agent_list)
    agent_val = pre_processing(agent_str)
    agent_sentiment, agent_sentiment_score = predict(agent_val)

    chatlog = pre_processing(sentence)
    chat_sentiment, chat_sentiment_score = predict(chatlog)
    print(agent_sentiment_score, chat_sentiment_score)

    # print("Agent sentiment:", agent_sentiment)
    # print("Chat sentiment:", chat_sentiment)

    agent_sentiment_list.append(agent_sentiment)
    chat_sentiment_list.append(chat_sentiment)
    agent_sentiment_score_list.append(agent_sentiment_score)
    chat_sentiment_score_list.append(chat_sentiment_score)

    # l.append(val)

# agent_sentiment_index = []
# for i, sentiment in zip(agent_sentiment_list):
#     agent_sentiment_index.append(sentiment)


b['Chat Sentiment'] = chat_sentiment_list
b['Agent Sentiment'] = agent_sentiment_list
b['Agent Sentiment Score'] = agent_sentiment_score_list
b['Chat Sentiment Score'] = chat_sentiment_score_list

b.head()
