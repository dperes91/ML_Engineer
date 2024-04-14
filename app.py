import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

#fname = 'arremessos_kobe.pkl'
dev_file = 'processed\prediction_test.parquet'
prod_file = 'processed\prediction_prod.parquet'

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

############################################ SIDE BAR TITLE
st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Análise dos Arremessos de Kobe Bryant.
""")

df_prod = pd.read_parquet(prod_file)
df_dev = pd.read_parquet(dev_file)

#st.write(df_dev)
#st.write(df_prod)

fignum = plt.figure(figsize=(10,4))

#plot df_dev.prediction_score_1 and df_prod.predict_score in the smae plot
plt.hist(df_dev.prediction_score_1, bins=50, alpha=0.5, color='blue')
plt.hist(df_prod.predict_score, bins=50, alpha=0.5, color='red')

#plt.plot(df_dev.prediction_score_1, color='blue')
#plt.plot(df_prod.predict_score, color='red')

# sns.displot(df_dev.prediction_score_1, color='blue')
# sns.displot(df_prod.predict_score, color='red')

#sns.displot(df_dev.prediction_score_1, kind='kde', color='blue', label='Development Data')
#sns.displot(df_prod.predict_score, kind='kde', color='red', label='Production Data')

# Plot
plt.xlabel('Prediction Score')
plt.ylabel('Density')
plt.title('Distribution of Prediction Scores')
plt.legend(['Desenvolvimento', 'Produção'])
# plt.xlabel('Probabilidade de Sucesso')
# plt.ylabel('Densidade')
# plt.grid(True)

st.pyplot()

st.write(metrics.classification_report(df_dev.shot_made_flag,
                                       df_dev.prediction_label
                                       ))