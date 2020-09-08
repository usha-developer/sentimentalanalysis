from flask import *
import sqlite3
import pickle
import numpy as np 
import pandas as pd 
#import seaborn as sns
import matplotlib.pyplot as plt
#sns.set_context("talk")
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import smtplib
server=smtplib.SMTP("smtp.gmail.com",587)
server.starttls()
server.login("subashpython1597@gmail.com","subash@1234")

con=sqlite3.connect('sentimentemotion.db')
cur=con.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS userdata(id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,name TEXT NOT NULL UNIQUE,email	TEXT NOT NULL UNIQUE,password	TEXT NOT NULL UNIQUE)')
cur.execute('CREATE TABLE IF NOT EXISTS textresult(id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,name TEXT NOT NULL,data TEXT NOT NULL, emotion TEXT NOT NULL,polarity TEXT NOT NULL)')


#sending mail


#emotion data code
data=pd.read_csv("emotion.data")
X = data['text']
y = data['emotions']
X_train,X_test,ytrain,ytest=train_test_split(X,y,test_size=0.10,random_state=42)
vectorizer = TfidfVectorizer( max_df= 0.9).fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
encoder = LabelEncoder().fit(ytrain)
y_train = encoder.transform(ytrain)
y_test = encoder.transform(ytest)
model = LogisticRegression(C=.1, class_weight='balanced')
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print("Training Accuracy : ", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy  : ", accuracy_score(y_test, y_pred_test))


#polarity code

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



app=Flask(__name__)
#secret_key='myworld'
@app.route('/')
def main():
    return render_template('home.html')



@app.route('/emotionprediction',methods=['POST'])
def saveemotion():
    if request.method=="POST":
        a=request.form['txt']
               
        tfidf = vectorizer.transform([a])
        preds = model.predict_proba(tfidf)[0]
        preds=list(preds)
        
        tfidf1 = vectorizer1.transform([a])
        preds1 = model1.predict_proba(tfidf1)[0]    
        #print(preds1)
        preds1=list(preds1)
        re1=preds1.index(max(preds1))

        d={0:'negative',1:'neutral',2:'positive'}
        re1=d[re1]
        re1=str(re1)

        re=preds.index(max(preds))
        if re==0:
            result1=re1
            result="anger"
            finalresult="emotion={0} and polarity={1}and mail={2}".format(result,result1,session['email'])
            con=sqlite3.connect('sentimentemotion.db')
            cur=con.cursor()
            #cur.execute('select * from userdata where name=?',(session['username'],))
            #row=cur.fetchone()
            #username=row[1]
            cur.execute('insert into textresult(name,data,emotion,polarity) values(?,?,?,?)',(session['username'],a,result,result1))
            con.commit()
            server.sendmail("subashpython1597@gmail.com","shifelfelix@gmail.com",finalresult)
            return render_template('emotionresult.html',temp=result,temp1=result1)
        elif re==1:
            result="fear"
            result1=re1
            finalresult="emotion={0} and polarity={1}".format(result,result1,session['email'])
            con=sqlite3.connect('sentimentemotion.db')
            cur=con.cursor()
            #cur.execute('select * from userdata where name=?',(session['username'],))
            #row=cur.fetchone()
            #username=row[1]
            cur.execute('insert into textresult(name,data,emotion,polarity) values(?,?,?,?)',(session['username'],a,result,result1))
            con.commit()
            server.sendmail("subashpython1597@gmail.com","shifelfelix@gmail.com",finalresult)
            return render_template('emotionresult.html',temp=result,temp1=result1)
            
        elif re==2:
            result="joy"
            result1=re1
            finalresult="emotion={0} and polarity={1}and mail={2}".format(result,result1,session['email'])
            con=sqlite3.connect('sentimentemotion.db')
            cur=con.cursor()
            
            #cur.execute('insert into textresult(name,data,emotion,polarity) values(?,?,?,?)',(session['username'],a,result,result1))
            #con.commit()
            server.sendmail("subashpython1597@gmail.com","shifelfelix@gmail.com",finalresult)
            return render_template('emotionresult.html',temp=result,temp1=result1)
            
            #print("joy")
        elif re==3:
            result="love"
            result1=re1
            finalresult="emotion={0} and polarity={1} and mail={2}".format(result,result1,session['email'])
            con=sqlite3.connect('sentimentemotion.db')
            cur=con.cursor()
            cur.execute('select * from userdata')
            row=cur.fetchone()
            username=row[1]
            #cur.execute('select * from userdata where name=?',(session['username'],))
            #row=cur.fetchone()
            #username=row[1]
            #cur.execute('insert into textresult(name,data,emotion,polarity) values(?,?,?,?)',(session['username'],a,result,result1))
            #con.commit()
            server.sendmail("subashpython1597@gmail.com","shifelfelix@gmail.com",finalresult)
            return render_template('emotionresult.html',temp=result,temp1=result1)
            
            #print("love")
        elif re==4:
            result="sadness"
            result1=re1
            finalresult="emotion={0} and polarity={1} and mail={2}".format(result,result1,session['email'])
            con=sqlite3.connect('sentimentemotion.db')
            cur=con.cursor()
            #cur.execute('select * from userdata where name=?',(session['username'],))
            #row=cur.fetchone()
            #username=row[1]
            cur.execute('insert into textresult(name,data,emotion,polarity) values(?,?,?,?)',(session['username'],a,result,result1))
            con.commit()
            server.sendmail("subashpython1597@gmail.com","shifelfelix@gmail.com",finalresult)
            return render_template('emotionresult.html',temp=result,temp1=result1)
            
            #print("saddness")
        elif re==5:
            result="surprise"
            #print("surprise")
            result1=re1
            finalresult="emotion={0} and polarity={1} and mail={2}".format(result,result1,session['email'])
            con=sqlite3.connect('sentimentemotion.db')
            cur=con.cursor()
            #cur.execute('select * from userdata where name=?',(session['username'],))
            #row=cur.fetchone()
            #username=row[1]
            #cur.execute('insert into textresult(name,data,emotion,polarity) values(?,?,?,?)',(session['username'],a,result,result1))
            #con.commit()
            server.sendmail("subashpython1597@gmail.com","shifelfelix@gmail.com",finalresult)
            return render_template('emotionresult.html',temp=result,temp1=result1)
        else:
            pass
    return render_template('emotion.html')

@app.route('/login.html')
def login1():
    return render_template('login.html')


@app.route('/register.html')
def reg1():
    return render_template('register.html')


@app.route('/loginpage',methods=['POST','GET'])
def savelog():
    if request.method=='POST':
        n1=request.form['name']
        n2=request.form['password']
        #session['username']=n1
        con=sqlite3.connect('sentimentemotion.db')
        cur=con.cursor()
        cur.execute('select * from userdata where name=?',(n1,))
        row=cur.fetchone()
        a1=row[1]
        a2=row[3]
        a3=row[2]
        #session['email']=rows[2]
        if a1==n1 and a2==n2:
            session['username']=a1
            session['email']=a3
            return render_template('emotion.html',name=session['username'],email=session['email'])
        else:
            return render_template('login.html')
    return render_template('login.html')
        

@app.route('/registration',methods=['POST','GET'])
def savereg():
    if request.method=='POST':
        uname=request.form['uname']
        #session[username]=uname
        uemail=request.form['uemail']
        upassword=request.form['upassword']
        upassword1=request.form['upassword1']
        if upassword==upassword1:
            with sqlite3.connect('sentimentemotion.db') as con:
                cur=con.cursor()
                cur.execute('insert into userdata(name,email,password) values(?,?,?)',(uname,uemail,upassword))
                con.commit()
                #session['email']
                return redirect('login.html')
        else:
            return render_template('register.html')
    return render_template('register.html')

@app.route('/emotion.html')
def emotion1():
    return render_template('emotion.html')

@app.route('/about.html')
def aboutus():
    return render_template('about.html')

@app.route('/home.html')
def aboutus1():
    return render_template('home.html')

'''@app.route('/sentimentemotion',methods=['POST','GET'])
def emotion2():
    if request.method=='POST':
        text=request.form['txt']'''
        

if __name__=='__main__':
    app.secret_key = 'myworld'
    app.run(debug=True)
