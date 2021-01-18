
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import re
import nltk
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


airline_tweets = pd.read_csv("Tweets.csv")


# In[4]:


airline_tweets.head()


# In[5]:


import seaborn as sns


# In[6]:


airline_tweets.info()


# In[7]:


airline_tweets.isnull().any()


# In[8]:


sns.barplot(x='airline_sentiment',y='airline_sentiment_confidence', data= airline_tweets)


# In[9]:


corr = airline_tweets.corr()


# In[13]:


sns.heatmap(corr)


# In[14]:


features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values


# In[15]:


processed_features = []
for sentence in range(0,len(features)):
    processed_feature = re.sub(r'\W',' ',str(features[sentence]))
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+',' ',processed_feature)
    processed_feature = re.sub(r'\s+',' ',processed_feature, flags=re.I)
    processed_feature= processed_feature.lower()
    processed_features.append(processed_feature)


# In[16]:


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer= TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train,y_test= train_test_split(processed_features, labels, test_size=0.2, random_state=0)


# In[19]:


X_train.shape


# In[20]:


from sklearn.ensemble import RandomForestClassifier


# In[21]:


text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)


# In[22]:


text_classifier.fit(X_train,y_train)


# In[23]:


predictions= text_classifier.predict(X_test)


# In[24]:


predictions


# In[25]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[26]:


print(accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

