import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import joblib
##########################################


######## Load Data ######
df= pd.read_csv('data\churn.csv')


#### Pipeline #########
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
    


def selection(df):
    df1 = df[["CreditScore", "Gender","Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", 
          "EstimatedSalary", 1, 2	,3,	4,'France',	'Germany'	,'Spain', "Exited"]] #"AgeCategory"
    return df
df = selection(df)
def splitting(df):
    y = df['Exited']
    X = df.drop(['Exited'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return  X_test, y_test 

#load the data sampled
X_train_sampled = pd.read_csv("data\X_train_sampled_data.csv")
y_train_sampled = pd.read_csv("data\y_train_sampled_data.csv")
#Getting the test data
X_test, y_test = splitting(df)

y_train_sampled = y_train_sampled["Exited"]

########################
#--------Models---------
########################

#### Model1 ---- Gradient Boosting Classifier 
GBC_final = GradientBoostingClassifier(learning_rate= 0.2, n_estimators=100, max_depth=15, max_features=12)
GBC_final.fit(X_train_sampled, y_train_sampled)
GBC_final.fit(X_train_sampled,y_train_sampled)
y_pred_GBC_final= GBC_final.predict(X_test)
print(f"Acccuracy: {accuracy_score(y_test, y_pred_GBC_final)*100}%")
print(f"Precision: {precision_score(y_test, y_pred_GBC_final)*100}%")
print(f"Recall:    {recall_score(y_test, y_pred_GBC_final)*100}%")
print(f"F1-Score:  {f1_score(y_test, y_pred_GBC_final)*100}%")

#### Model2 ---- Extra Tree Classifier 
EXC_final = ExtraTreesClassifier(criterion="gini", n_estimators=100, max_depth=15, max_features=10)
EXC_final.fit(X_train_sampled, y_train_sampled)

EXC_final.fit(X_train_sampled, y_train_sampled) 
y_pred_EXC_final = EXC_final.predict(X_test)
print(f"Acccuracy: {accuracy_score(y_test, y_pred_EXC_final)*100}%")
print(f"Precision: {precision_score(y_test, y_pred_EXC_final)*100}%")
print(f"Recall:    {recall_score(y_test, y_pred_EXC_final)*100}%")
print(f"F1-Score:  {f1_score(y_test, y_pred)*100}%")


# model_FV = {'model': model_final, 'features_modelo': list(X_train.columns)}