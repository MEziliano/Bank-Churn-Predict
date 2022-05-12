import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import streamlit as st


df= pd.read_csv('churn.csv')


def treate_data(df):
  df['AgeGroup'] = df['Age'].apply(lambda x: 1 if x >=55  else 0)  
  df["AgeCategory"] = 'Adult'
  df.loc[df['Age'] >= 55, 'AgeCategory'] ='Senior'
  df.loc[df['Age'] <= 25, 'AgeCategory'] = 'Young'
  df['Gender']= df['Gender'].map({'Male':0, 'Female':1})
  return df

def get_dummies(df):
  Products = pd.get_dummies(df['NumOfProducts'])
  Geo = pd.get_dummies(df['Geography'])
  df = pd.concat([df, Products, Geo], axis=1)
  return df

def delete(df):
  df = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Geography', 'AgeCategory'], axis=1)
  return df

df = (df.
    pipe(treate_data).
    pipe(get_dummies).
    pipe(delete))
    

st.set_page_config(layout="wide")

@st.cache()

def selection(df):
    df1 = df[["CreditScore", "Gender","Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", 
          "EstimatedSalary", 1, 2	,3,	4,'France',	'Germany'	,'Spain', "Exited"]] #"AgeCategory"
    return df

def splitting(df):
    y = df1['Exited']
    X = df1.drop(['Exited'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X,y

df = selection(df)
