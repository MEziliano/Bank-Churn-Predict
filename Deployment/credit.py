import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# --------------------------
#  Streamlit application
# --------------------------
# Step 1 - load the data
df= pd.read_csv('transactional-sample.csv')

# Step 2 — Clean the messing data 
def treate_date(df):
  df['transaction_date'] = df['transaction_date'].str.replace('T', '-')
  x = df['transaction_date'].str.split('-', n=3, expand=True)
  df[['Year', 'Month', 'day', 'time']]=x
  y = df['time'].str.split(':', n=3, expand=True)
  df[['Hour', 'Min', 'ssec']] = y 
  z = df['ssec'].str.split('.', n=2, expand=True)
  df[['sec', 'nsec']] = z
  a = df['card_number'].str.split('*', n=1, expand=True)
  df[['card_begin', 'card_end']] = a
  df['card_end'] = df['card_end'].str.replace('*', '')
  return df

# That function will be important for the second part of the project 
# Step 3 — Cleaning part II — The Return of messgin data
def treate_data(df):
  
  df['device_id'] = df['device_id'].fillna(0)
  df['has_cbk'] = df['has_cbk'].map({False: "Good", True: "Chargeback"})
  return df 

# Step 4 — Data Transformation 
def transform(df):
  df['card_begin'] = df['card_begin'].astype(int)
  df['card_end'] = df['card_end'].astype(int)
  df['Month'] = df['Month'].astype(int)
  df['day']= df['day'].astype(int)
  df['Hour']= df['Hour'].astype(int)
  df['Min'] = df['Min'].astype(int)
  return df


# df = load().treate_date().treate_data().tranform()
(df.pipe(treate_date)
   .pipe(treate_data)
   .pipe(transform)
 )
st.set_page_config(layout="wide")

@st.cache()
def new_func(df):
    df = df[['transaction_id', 'merchant_id', 'user_id', 'card_begin','card_end',	'Month', 'day','Hour', 'transaction_amount', 'device_id', 'has_cbk']]
    return df
df = new_func(df)
# --------------------------
#  Streamlit application
# --------------------------

 
st.title("CloudWalk - Transactions Analysis ")
st.header("Analysis of November and Dezember of 2019")
st.markdown("""## Contextualization
### 1. The acquirer market
Since the begin of the humankind exchanges exists. But in last few years, none setor of the economics has changes as the Credit Card bank or simply Acquirer. Banks and payment networks were invented in a pre-internet era and since then have been incrementally evolving. 
The flow, both money and information, contains the participition of many agents. Which means, it's doesn't allways it's fair with the custumers. Agents like:
* Payment Geteway;
* Acquirer;
* Sub-acquirer;
* Bank issues, and etc.""") 

st.markdown("""### 2. What's is an acquirer?
A gateway (also called payment gateway) is a system that transmits data from purchases made in your store at checkout to companies, the gateway acts as a terminal, integrating in all the transactions carried out between the players of the payment flow in a single place.
An acquirer is a company that specializes in processing payments. A sub-acquirer is a company that processes payments and transmits the generated data to the other players involved in the [payment flow](https://help.vtex.com/tutorial/what-is-the-difference-between-acquirer-brand-gateway-and-sub-acquirer-in-brazil--1dyPJ3gQCCO4ea2o6OMgCi) 
### 3. Chargeback 
A chargeback, occurs when a cardholder questions a transaction and asks their card-issuing bank to reverse it. If a cardholder sees a charge from your business but never bought anything from you, it could mean fraud is at play. Both the chargeback as the cancellation are tools to make the system more efficient against fraud. """)

st.write("Now, let's check the secont part of the challange")
st.markdown("A few comments about the data. The columns of card number and transaction date has to pass to some changes. As we can see:")

st.title("Credit Analysis")
df_raw = pd.read_csv("transactional-sample.csv")
if st.checkbox("The Dataset", False):
    st.write(df_raw)
st.write("This is our dataset after some treatemnt")
if st.checkbox("The Dataset after some treatement", False):
    st.write(df)

st.header("Proportion of Chargeback at the period")
fig = px.pie(df['has_cbk'], names=df['has_cbk'], title='Charge Back') 
st.plotly_chart(fig)

st.header("Verifying the consum habits in the period")
fig1 = px.bar(df, x=df['transaction_amount'], y=df['day'], color=df['has_cbk'], orientation='h', title="Transction over the month")
fig.update_xaxes(title_text='Days')
fig.update_yaxes(title_text='Transactions')
st.plotly_chart(fig1)

fig2 = px.density_heatmap(df, x=df['day'], y=df['transaction_amount'], marginal_x="histogram", marginal_y="histogram", title="Density of Transactions by day")
fig.update_xaxes(title_text='Days')
fig.update_yaxes(title_text='Count')
st.plotly_chart(fig2)

fig3 = px.density_heatmap(df, x=df['Hour'], y=df['transaction_amount'], marginal_x="histogram", marginal_y="histogram", title="Transaction by Hours")
fig.update_xaxes(title_text='Hour')
fig.update_yaxes(title_text='Count')
st.plotly_chart(fig3)

st.header("Verifying the transaction by the users") 
fig4 = px.histogram(df,x=df['user_id'],
                   title='Transactions x Users', color=df['has_cbk'])
fig.update_xaxes(title_text='Users')
st.plotly_chart(fig4)

fig5 = px.histogram(df,x=df['merchant_id'],
                   title='Transactions by Merchant', color=df['has_cbk'])
fig.update_xaxes(title_text='Merchant')
fig.update_yaxes(title_text='Count')
st.plotly_chart(fig5)

fig6 = px.histogram(df,x=df['card_begin'],
                   title='Transactions across the month', color=df['has_cbk'])
fig.update_xaxes(title_text='Card number')
fig.update_yaxes(title_text='Count')
fig.update_layout(showlegend=False)
st.plotly_chart(fig6)

st.title("Conclusions")
st.markdown("""According to the dataset analyzed in the period, we can conclude that the presence of chargebacks is not assumed based on the day or time of the transactions.

A recommendation for the dataset would be to include location data, given that all transactions were made from mobile devices.""")
