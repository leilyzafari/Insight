#!/usr/bin/env python
# coding: utf-8

# ## Identify topics covered in each review

# Importing libraries and data

# In[3]:


import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
#from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import seaborn as sns; sns.set()

from imblearn.over_sampling import RandomOverSampler,SMOTE


# In[90]:


# Import dataset 
df_data_reviews = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')


# In[91]:


df_data_reviews.head()


# In[10]:


df_data_reviews.shape


# In[11]:


def label_sentiment(row):
    if row['Liked']==1:
        return 'positive'
    elif row['Liked']==0:
        return 'negative'
    elif row['Liked']==2:
        return 'neutral'
    else:
        return  row['Liked']


# Run the Topic Modelling function on the data

# In[12]:


df_data_reviews['Liked'] = df_data_reviews.apply (lambda row: label_sentiment(row), axis=1)


# In[13]:


df_data_reviews.head()


# In[14]:


df_data_reviews['Review'] = df_data_reviews.apply (lambda row: row.str.lower(), axis=1)


# In[15]:


# Utility Functions
import re
from nltk.stem.porter import PorterStemmer
bad_chars = set(["@", "+", '<br>', '<br />', '/', "'", '"', '\\',
                        '(',')', '<p>', '\\n', '<', '>', '?', '#', ',',
                        '.', '[',']', '%', '$', '&', ';', '!', ';', ':',
                        '-', "*", "_", "=", "}", "{","!"])
def default_clean(text):
    '''
    Removes default bad characters
    '''
    #if text!='':
    #text = filter(lambda x: x in string.printable, text)
    
    #print (text)
    for char in bad_chars:
            text = str(text).replace(char, " ")
    text = re.sub('\d+', "", str(text))
    return text

def stop_and_stem(text, stem=True, stemmer = PorterStemmer()):
    '''
    Removes stopwords and does stemming
    '''
    stoplist = stopwords.words('english')
    #stoplist.extend(['meal','first','table','always','toronto','restaurant','years','never'])
    if text is None:
        return
    text_stemmed=[]
    if stem:
        for document in text:
            if document is not None:
                for word in document.split():
                    if word not in stoplist:
                        text_stemmed.append(stemmer.stem(word))
        #text_stemmed = [[stemmer.stem(word) for word in document.split()
        #                 if word not in stoplist] for document in text]
    else:
        text_stemmed = [[word for word in document.split()
                 if word not in stoplist] for document in text]

    return text_stemmed


# In[92]:


df_data_reviews['Review-Clean']=df_data_reviews['Review']
df_data_reviews.head()


# In[93]:


df_data_reviews['Review-Clean'] = df_data_reviews['Review-Clean'].apply (lambda x: default_clean(x))


# In[94]:


#df_data_reviews['Review-Clean'] = df_data_reviews.apply (lambda row: stop_and_stem(row), axis=1)


# In[95]:


df_data_reviews.head()


# In[96]:


df_data_reviews.groupby('Liked')['Liked'].count().plot(kind = 'bar')


# Create training and validation set

# In[97]:


X_train, X_test, y_train, y_test = train_test_split(df_data_reviews['Review'],
                                                    df_data_reviews['Liked'],
                                                    test_size=0.25,
                                                    random_state=0,
                                                   shuffle=True)


# In[98]:


y_train.head()


# Oversampling and Building the Model

# In[99]:


ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train.to_frame(), y_train)

X_resampled = pd.Series(X_resampled.iloc[:,0]); y_resampled = pd.Series(y_resampled)


# In[100]:


import re,string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# In[101]:


# Utility Functions
def default_clean(text):
    '''
    Removes default bad characters
    '''
    #if text!='':
    #text = filter(lambda x: x in string.printable, text)
    
    bad_chars = set(["@", "+", '<br>', '<br />', '/', "'", '"', '\\',
                        '(',')', '<p>', '\\n', '<', '>', '?', '#', ',',
                        '.', '[',']', '%', '$', '&', ';', '!', ';', ':',
                        '-', "*", "_", "=", "}", "{"])
    for char in bad_chars:
            text = text.replace(char, " ")
    print("text",text)
    text = re.sub('\d+', "", str(text))

    return text

def stop_and_stem(text, stem=True, stemmer = PorterStemmer()):
    '''
    Removes stopwords and does stemming
    '''
    stoplist = stopwords.words('english')
    stoplist.extend(['meal','first','table','always','toronto','restaurant','years','never'])
    if stem:
        text_stemmed = [[stemmer.stem(word) for word in document.lower().split()
                         if word not in stoplist] for document in text]
    else:
        text_stemmed = [[word for word in document.lower().split()
                 if word not in stoplist] for document in text]

    return text_stemmed


# In[102]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
# from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
# stoplist = stopwords.words('english')

model = Pipeline([('vect', CountVectorizer(ngram_range = (1,2))),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
])


# In[103]:


print(X_resampled.values.ravel(), type(X_resampled.values),X_resampled.values.astype('U'), type(X_resampled.values.astype('U')))
print(y_resampled.values.astype('U'), type(y_resampled.values.astype('U')))


# In[104]:


model.fit(X_resampled.values.astype('U').ravel(), y_resampled.values.astype('U'))


# In[105]:


yhat_test = model.predict(X_test.values.astype('U'))


# In[106]:


from sklearn.metrics import confusion_matrix
import numpy
unique, counts = numpy.unique(yhat_test.astype('U') ,return_counts=True)
mat = confusion_matrix(y_test.astype('U'),yhat_test.astype('U'))
mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(mat.T, square=True, annot=True, fmt='.2f', cmap=plt.cm.Blues,
            cbar=False,xticklabels=['Negative','Neutral','Positive'], yticklabels=['Negative','Neutral','Positive'])


# In[107]:


df_confusion = pd.crosstab(pd.Series(y_test.reset_index(drop=True),name='Actual'),
                           pd.Series(yhat_test,name='Predicted'))
df_confusion


# In[108]:


accuracy = np.mean(yhat_test==y_test)
print (accuracy)


# In[74]:


from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
# Learn to predict each class against the other
oneVsRestClassifier = OneVsRestClassifier(model)

y_binary = label_binarize(df_data_reviews['Liked'], classes=['positive', 'neutral', 'negative'])

X_train, X_test, y_train, y_test = train_test_split(df_data_reviews['Review'],
                                                    df_data_reviews['Liked'],
                                                    test_size=0.25,
                                                    random_state=0)                                                   
# Resaample
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train.to_frame(), y_train)

X_resampled = pd.Series(X_resampled.iloc[:,0])
y_resampled = pd.Series(y_resampled)

y_test_binary = label_binarize(y_test, classes=['positive', 'neutral', 'negative'])
oneVsRestClassifier.fit(X_resampled.values.astype('U'), y_resampled)

y_predicted =oneVsRestClassifier.predict(X_test)
y_predicted_binary = label_binarize(y_predicted, classes=['positive', 'neutral', 'negative'])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_predicted_binary[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
sentiment_list=[]
for sentence in X_test:
    score = analyser.polarity_scores(sentence)
    if score['compound']>0.5:
        sentiment_list.append('positive')
    elif score['compound']<-0.5:
        sentiment_list.append('negative')
    else:
        sentiment_list.append('neutral')
    
vader_binary = label_binarize(sentiment_list, classes=['positive', 'neutral', 'negative'])
vader_fpr = dict()
vader_tpr = dict()
roc_auc = dict()
for i in range(3):
    vader_fpr[i], vader_tpr[i], _ = roc_curve(y_test_binary[:, i], vader_binary[:, i])
    roc_auc[i] = auc(vader_fpr[i], vader_tpr[i])
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='green',label='NB')
plt.plot(vader_fpr[2], vader_tpr[2], color='orange',label='Vader')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()


# In[75]:


def predict_category(s, train=y_train, model=model):
    pred = model.predict([s])
    return pred[0]


# In[76]:


predict_category("I went there with my husband")


# In[77]:


predict_category("There is music.")


# In[78]:


predict_category("Excellent charm and wonderful seller")


# In[79]:


predict_category("I was not pleased with the quality of the product")


# In[80]:


df_confusion.loc['negative','negative'] / df_confusion.loc['negative',:].sum()


# In[81]:


precision_positive = df_confusion.loc['positive','positive'] / df_confusion.loc[:,'positive'].sum()
precision_negative = df_confusion.loc['negative','negative'] / df_confusion.loc[:,'negative'].sum()
precision_neutral = df_confusion.loc['neutral','neutral'] / df_confusion.loc[:,'neutral'].sum()
precision_positive + precision_negative+precision_neutral


# In[82]:


recall_positive = df_confusion.loc['positive','positive'] / df_confusion.loc['positive',:].sum()
recall_negative = df_confusion.loc['negative','negative'] / df_confusion.loc['negative',:].sum()
recall_neutral = df_confusion.loc['neutral','neutral'] / df_confusion.loc['neutral':,].sum()
recall_positive + recall_negative+recall_neutral


# In[89]:


dataset = pd.read_csv('reviews_jackastor.csv', delimiter = ',')
testSentenceList = []
sentimentList = []
from nltk.data import sent_tokenize
for row in dataset['text']:
    text = row.replace('.','. ')
    for sentence in sent_tokenize(text):
        if len(sentence)<2:
            continue
        testSentenceList.append(sentence)
series=pd.Series(testSentenceList).astype(str)
predicted = model.predict(series)


# In[ ]:


pred_df = pd.DataFrame(
    {'Sentences': series,
     'sentiment':predicted
    })
pred_df.to_csv("reviews_sentiment_NB_blu_opt_1.csv")


# In[85]:



import joblib
## dump the model
joblib.dump(model, 'sentiment-analysis.pkl')


# In[ ]:




