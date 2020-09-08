import sklearn
import numpy as np
import pandas as pd
##import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
data1=pd.read_csv("Twitter_Data.csv")
data1.isnull().sum()
data1["input"]=data1["clean_text"].fillna("hi")
data1["output"]=data1["category"].fillna(0.0)
X1 = data1['input']
y1 = data1['output']
X1_train,X1_test,y1train,y1test=train_test_split(X1,y1,test_size=0.10,random_state=42)
vectorizer1 = TfidfVectorizer( max_df= 0.9).fit(X1_train)
X1_train = vectorizer1.transform(X1_train)
X1_test = vectorizer1.transform(X1_test)
encoder1 = LabelEncoder().fit(y1train)
y1_train = encoder1.transform(y1train)
y1_test = encoder1.transform(y1test)
model1 = LogisticRegression(C=.1, class_weight='balanced')
model1.fit(X1_train,y1_train)
y1_pred_train = model1.predict(X1_train)
y1_pred_test = model1.predict(X1_test)
print("Training Accuracy : ", accuracy_score(y1_train, y1_pred_train))
print("Testing Accuracy  : ", accuracy_score(y1_test, y1_pred_test))
##def predict_(x, plot=False):
##    tfidf = vectorizer.transform([x])
##    preds = model.predict_proba(tfidf)[0]
##    plt.figure(figsize=(6,4))
##    sns.barplot(x= encoder.classes_, y=preds)
##    plt.show()
##    return preds
aa=input("Enter the Message:")
tfidf1 = vectorizer1.transform([aa])
preds1 = model1.predict_proba(tfidf1)[0]    
print(preds1)
