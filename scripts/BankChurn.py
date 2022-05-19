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
import pickle

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
st.set_page_config(layout="wide")
@st.cache()
##################################################
# Loading the data
##################################################
def load_data():
  return  pd.read_csv("data\churn.csv")

df = load_data()

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

##################################################
# Configurating the pipeline
##################################################

df = (df.pipe(treate_data)
.pipe(get_dummies)
.pipe(delete))

def select(df):
    df = df[["CreditScore", "Gender","Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", 1, 2,3, 4,'France', 'Germany','Spain', "Exited"]]
    return df

df = select(df)

##################################################
# Final adjusted 
##################################################
y = df.Exited 
x = df.drop(columns=["Exited"])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)
##################################################
# Getting the models
##################################################
# Extra Tree
EXC_in = open('EXC_model.pickle', 'rb') 
model_EXC = pickle.load(EXC_in)
# Gradient Boosting
GBC_in = open('GBC_model.pkl', 'rb') 
model_GBC = pickle.load(GBC_in)

##################################################
# Create a checkbox to show the dataset
##################################################
if st.sidebar.checkbox("Check the costumers", False):
  st.subheader("Veja os Equipamentos")
  st.write(df)

##################################################
# Function to visualizes the metrics 
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

################################################################################################################
# --------------------------------------------- Criando as Máquinas Preditivas - Classificadores ----------------------------------------------------
################################################################################################################

st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.subheader("Chose the Machine")
classifier = st.sidebar.selectbox("Classifier", ("Extra Tree Classifier", "Gradient Boost Classifier"))

##################################################
# MP 1 - Extra Tree Classifier
##################################################

if classifier == "Extra Tree Classifier":
    metrics = st.sidebar.multiselect("Qual métrica utilizar?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Extra Tree Classifier results")
        model = DecisionTreeClassifier(criterion="entropy")
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test) 
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
        plot_metrics(metrics)
if classifier == "Random Forest":
    st.sidebar.subheader("Hyperparâmetros")
    n_estimators= st.sidebar.number_input("O n° de árvores de decisão na floresta", 100, 5000, step=10, key="n_estimators")
    max_depth = st.sidebar.number_input("O mx_depth of tree - Profundidade da Árvore", 1, 20, step =1, key="max_depth")
    bootstrap = st.sidebar.radio("Amostras - Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
    
    metrics = st.sidebar.multiselect("Qual métrica utilizar?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap= bootstrap, n_jobs=-1 )
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

##################################################
# MP 1 - Gradient Boost Classifier
##################################################

if classifier == "Gradient Boost Classifier":
    metrics = st.sidebar.multiselect("Qual métrica utilizar?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Gradient Boost Classifier")
        model = MLPClassifier(hidden_layer_sizes=(100, 50, 10), max_iter=1000)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
    
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)
####################################################################################################
# Criando a função que irá fazer a predição usando os dados impostados pelo usuário do Sistema 
####################################################################################################
# Chanege for columns at that dataset: 
# 'CreditScore', 'Gender', 'Age','Tenure', 'Balance', 'NumOfProducts','HasCrCard', 'IsActiveMember' 'EstimatedSalary','AgeGroup', 1, 2, 3, 4,'France','Germany','Spain'
def prediction(Gender,Geography, renda, emprestimo, historico_credito):
    # Pre-processando a entrada do Usuário 
    # Sexo   
    if Gender == "Male":
        Gender = 0
    else:
        Gender = 1
    if Geography == "Germany":
        Geography = "France"
    else:
        
        estado_civil = 1
    # Histórico de crédio 
    if historico_credito == "Débitos Pendentes":
        historico_credito = 0
    else:
        historico_credito = 1  
 
    emprestimo = emprestimo / 1000
 
    # Fazendo Predições
    prediction = GBC_in.predict( 
        [[Gender, Geography, renda, emprestimo, historico_credito]])
     
    if prediction == 0:
        pred = 'Rejeitado'
    else:
        pred = 'Aprovado'
    return pred

# Essa função é para criação da webpage  
def main():
    # Elementos da webpage
    # Nesse Ponto vc deve Personalizar o Sistema com sua Marca
    html_temp = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:white;text-align:center;">SAE</h1> 
    <h2 style ="color:white;text-align:center;">Sistema de Aprovação de Empréstimos - by Edu</h2> 
    </div> 
    """
    #https://www.reuters.com/business/wall-street-banks-staff-churn-double-this-year-after-bonus-payouts-experts-2022-03-14/
    #https://www.acuitykp.com/blog/a-data-driven-approach-to-reduce-churn-in-financial-institutions/
      
    # Função do stramlit que faz o display da webpage
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # As linhas abaixo criam as caixas na qual o usuário vai entrar com dados da pessoa que quer o empréstimo para fazer a Predição
    sexo = st.selectbox('Sexo',("Masculino","Feminino"))
    estado_civil = st.selectbox('Estado Civil',("Solteiro(a)","Casado(a)")) 
    renda = st.number_input("Renda Mensal") 
    emprestimo = st.number_input("Valor Total do Empréstimo")
    historico_credito = st.selectbox('Histórico de Créditos',("Sem Débitos","Débitos Pendentes"))
    result =""
      
    #Quando o Usuário clicar no botão "Verificar" a Máquina Preditiva faz seu trabalho
    if st.button("Verificar"): 
        result = prediction(sexo, estado_civil, renda, emprestimo, historico_credito) 
        st.success('O empréstimo foi {}'.format(result))
        print(emprestimo)
#############################################################################
#-------------------------------------------------------- Web Desing develop 
#############################################################################




#Don't forget to run the conde at the terminal
# streamlit run your_script.py [-- script args]