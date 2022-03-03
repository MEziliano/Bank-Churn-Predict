#pacotes importantes
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

#funcoes
def predict(modelo, dataframe):
    dataframe[modelo['variaveis_scalling']] = modelo['scaler'].transform(dataframe[modelo['variaveis_scalling']])
    dataframe[modelo['variaveis_categoricas']] = modelo['encoder'].transform(dataframe[modelo['variaveis_categoricas']])
    dataframe['predicao'] = modelo['model'].predict(dataframe[modelo['features_modelo']])
    return dataframe

#Lendo os dados
teste_df = pd.read_csv('teste.csv', index_col = 0)
train_df = pd.read_csv('treino.csv', index_col = 0)
modelo = joblib.load('modelo.pickle')
dataframe_predicao  = predict(modelo, teste_df)

#criando o app
st.image('dnc.png')
st.header('DNC Day 1')
st.markdown('https://docs.streamlit.io/en/stable/api.html')
opcao = st.selectbox( 'Selecione a opcao desejada', ('Analise Exploratoria', 'Resultados do modelo'))
if opcao == 'Analise Exploratoria':
    #agora estamos dentro de analise exploratoria
    st.markdown('Desenvolva seu codigo. Exemplo: ')
    grafico = .distplot(dataframe_predicao['Age'])
    st.write(grafico.figure)
if opcao == 'Resultados do modelo':
    #agora estamos dentro de resultados do modelo
    st.markdown('Desenvolva seu codigo. Exemplo: ')
    st.markdown(classification_report(y_true = dataframe_predicao['Exited'], y_pred = dataframe_predicao['predicao']))



