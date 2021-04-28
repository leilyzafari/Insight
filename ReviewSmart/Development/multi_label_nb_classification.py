#!/usr/bin/env python
# coding: utf-8

# In[3]:


import xml.etree.ElementTree as ET
tree = ET.parse('dataset/Restaurants_Train.xml')
root = tree.getroot()


# In[4]:


# Use this dataframe for multilabel classification
# Must use scikitlearn's multilabel binarizer
import pandas as pd
labeled_reviews = []
for sentence in root.findall("sentence"):
    entry = {}
    aterms = []
    aspects = []
    if sentence.find("aspectTerms"):
        for aterm in sentence.find("aspectTerms").findall("aspectTerm"):
            aterms.append(aterm.get("term"))
    if sentence.find("aspectCategories"):
        for aspect in sentence.find("aspectCategories").findall("aspectCategory"):
            aspects.append(aspect.get("category"))
    entry["text"], entry["terms"], entry["aspects"]= sentence[0].text, aterms, aspects
    labeled_reviews.append(entry)
labeled_df = pd.DataFrame(labeled_reviews)
print("there are",len(labeled_reviews),"reviews in this training set")
labeled_df.head()


# In[59]:


import csv
#annotated_reviews_df = pd.read_csv("reviews_annotated_manual.csv",delimiter=";")
with open('annotated_reviews_df.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file,delimiter=";")
    line_count = 0
    labeled_reviews = []
    entryDict = {}
    for row in csv_reader:
        categoryList = []
        if row['text'] not in entryDict:
            entryDict[row['text']]=[]
        category = row['aspects'].strip()
        entryDict[row['text']].append(category)

for key,value in entryDict.items():
    entry={}
    entry['text']=key
    entry['aspects']=value
    labeled_reviews.append(entry)
print("there are",len(labeled_reviews),"reviews in this training set")   
labeled_df = pd.DataFrame(labeled_reviews)
labeled_df.head()


# In[6]:


# Save annotated reviews
labeled_df.to_pickle("annotated_reviews_df.pkl")
labeled_df.to_csv("annotated_reviews_df.csv")
labeled_df.head()


# In[7]:


annotated_reviews_df = pd.read_pickle("annotated_reviews_df.pkl")


# In[8]:


annotated_reviews_df.head()


# In[9]:


annotated_reviews_df =labeled_df
annotated_reviews_df.head()


# In[10]:


#annotated_reviews_df.shape
annotated_reviews_df['aspects'] = annotated_reviews_df['aspects'].apply(tuple)
#annotated_reviews_df.groupby(['aspects'])['text'].count().plot.bar(ylim=0)
valueList = []
valueDict = {}
for index,row in annotated_reviews_df.iterrows():
        for values in row['aspects']:
            #print (values)
            #value = values.replace("'",'').replace("'",'').replace('"','').replace('(','').replace(')','').strip()
            for value in values.split(','):
                if value!='':
                    valueList.append(value.strip())

for value in set(valueList):
    valueDict[value]=valueList.count(value)
print (valueDict.keys())
import matplotlib.pyplot as plt
from matplotlib import pyplot
#pl = pred_df.groupby('pred_category')['text_pro'].count().plot.barh()
plt.barh(list(valueDict.keys()), list(valueDict.values()))
#fig, ax = plt.subplots()
#ax.barh(list(valueDict.keys()), list(valueDict.values()))


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

# Convert the multi-labels into arrays
mlb = MultiLabelBinarizer()


# In[12]:


#aspects = annotated_reviews_df.aspects.to_frame('new_aspects')
y = mlb.fit_transform(annotated_reviews_df.aspects)
#y=annotated_reviews_df.aspects.to_frame()
X = annotated_reviews_df.text

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
# save the the fitted binarizer labels
# This is important: it contains the how the multi-label was binarized, so you need to
# load this in the next folder in order to undo the transformation for the correct labels.
filename = 'mlb.pkl'
pickle.dump(mlb, open(filename, 'wb'))


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import LabelPowerset
import numpy as np

# LabelPowerset allows for multi-label classification
# Build a pipeline for multinomial naive bayes classification
text_clf = Pipeline([('vect', CountVectorizer(stop_words = "english",ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', LabelPowerset(MultinomialNB(alpha=1e-1))),])
text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print(np.mean(predicted == y_test))
print(type(X_train))
print(type(y_train))


# In[14]:


from sklearn.model_selection import cross_validate,KFold
cv_scores = cross_validate(estimator=text_clf,
                              X=X_test,
                              y=y_test,
                              cv=KFold(shuffle=True, n_splits=5))
cv_scores


# In[10]:


from sklearn.metrics import hamming_loss,accuracy_score,precision_score,recall_score,classification_report
hamming_loss(predicted, y_test)
accuracy_score(predicted, y_test)
precision_score(y_test,predicted,average='macro')
precision_score(y_test,predicted,average='micro')
recall_score(y_test,predicted, average='micro')
recall_score(y_test,predicted, average='macro')
print (classification_report(y_test,predicted))
print (hamming_loss(predicted, y_test))


# In[24]:


# Test if SVM performs better
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', LabelPowerset(
                             SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, max_iter=6, random_state=42,tol=None)))])
text_clf_svm = text_clf_svm.fit(X_train, y_train)
predicted_svm = text_clf_svm.predict(X_test)

#Calculate accuracy
np.mean(predicted_svm == y_test)


# In[16]:


from sklearn.metrics import hamming_loss
hamming_loss(predicted_svm, y_test)


# In[17]:


dataset = pd.read_csv('reviews_scaramouche.csv', delimiter = ';')
predicted = text_clf.predict(dataset['text'])
pred_df = pd.DataFrame(
    {'text_pro': dataset['text'],
     'pred_category': mlb.inverse_transform(predicted)
    })
pred_df.head()


# In[58]:


from textblob import TextBlob
"""import nltk
nltk.download()
from nltk.tokenize import sent_tokenize"""
import spacy
nlp = spacy.load('en_core_web_sm')
review_list=[]
sentiment_list = []
for index,row in dataset.iterrows():
    for sentence in list(nlp(row['text']).sents):
        review_list.append(sentence.text)
        sentiment_list.append(TextBlob(sentence.text).sentiment.polarity)
series=pd.Series(review_list).astype(str)
predicted = text_clf.predict(series)
sentiment_series = pd.Series(sentiment_list).astype(str)


# In[52]:


pred_df = pd.DataFrame(
    {'text_pro': series,
     'pred_category': mlb.inverse_transform(predicted),
     'sentiment':sentiment_series
    })
pred_df.to_csv("reviews_annotated_version3_clean.csv")


# In[53]:


valueList = []
valueDict = {}
for index,row in pred_df.iterrows():
    values = row['pred_category']
    valueList.append(values)
valueList1=[]
for values in valueList:
    for value in values:
        for category in value.split(','):
            if category!='': 
                valueList1.append(category.strip())
for value in set(valueList1):
    valueDict[value]=valueList1.count(value)


# In[54]:


import matplotlib.pyplot as plt#, mpld3
from matplotlib import pyplot
pl = pred_df.groupby('pred_category')['text_pro'].count().plot.barh()

fig, ax = plt.subplots()
ax.barh(list(valueDict.keys()), list(valueDict.values()))
# mpld3.fig_to_html(fig,'plot.png')
#mpld3.save_html(fig,'plot.html')
# ax.figure.savefig('file.png')


# In[57]:


import joblib
joblib.dump(text_clf, 'model_mlb.pkl') #save the good model
joblib.dump(mlb, 'mlb.pkl') # 


# In[ ]:




