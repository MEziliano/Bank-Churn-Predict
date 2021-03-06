##################################################
# Importando Bibliotecas/Pacotes
##################################################

# Manipulating data 
import pandas as pd
import numpy as np

# visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 

# Web aplication
import streamlit as st

# Serializing and open models 
import joblib


from sklearn import metrics



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve,  plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

import sys




##################################################
# Criando a Aplicação no Streamlit
##################################################
@st.cache(allow_output_mutation=True, persist=False)
##################################################
# Importando os Dados
##################################################

def load():
    data= pd.read_csv('churn.csv')
    return data
df = load()

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

  sys.maxsize
  

df = (df.
    pipe(treate_data).
    pipe(get_dummies).
    pipe(delete))



##################################################
# Criando um checkbox para mostrar o Dataset
##################################################

if st.sidebar.checkbox("Ver o Dataset", False):
    st.subheader("Veja os Equipamentos")
    st.write(df)


##################################################
# Processando os dados - Amostragem
##################################################

@st.cache(persist=True, allow_output_mutation=True)

def split(df):
    y = df.target 
    x = df.drop(columns=["target"])
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)
    
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = split(df)


##################################################
# Função de Visualização de Métricas
##################################################

def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=   class_names)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
class_names = ["Em Operação", "Falha"]

##################################################
# Criando as Máquinas Preditivas - Classificadores
##################################################


st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.subheader("Escolha o Modelo")
classifier = st.sidebar.selectbox("Classifier", ("Decision Tree", "Random Forest", "Redes Neurais" ))

##################################################
# MP 1 - Decision Tree
##################################################

if classifier == "Decision Tree":
    metrics = st.sidebar.multiselect("Qual métrica utilizar?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Decision Tree results")
        model = DecisionTreeClassifier(criterion="entropy")
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
        plot_metrics(metrics)

##################################################
# MP 1 - Redes Neurais
##################################################

if classifier == "Redes Neurais":
    metrics = st.sidebar.multiselect("Qual métrica utilizar?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Network Results")
        model = MLPClassifier(hidden_layer_sizes=(100, 50, 10), max_iter=1000)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
    
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

##################################################
# MP 2 - Random Forest
##################################################

if classifier == "Random Forest":
    metrics = st.sidebar.multiselect("Qual métrica utilizar?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=200, max_depth=2, )
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)


#Don't forget to run the conde at the terminal
# streamlit run your_script.py [-- script args]