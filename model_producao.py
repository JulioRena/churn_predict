import pandas as pd
from tratamento import *
import joblib

new_df = pd.read_csv("datasets/Abandono_teste.csv", sep=';')
df_resultado = new_df.copy()

pipeline = joblib.load('modelo_treinado.pkl')

remover_variavel(new_df, 'RowNumber')
remover_variavel(new_df, 'CustomerId')
remover_variavel(new_df, 'Surname')
remover_variavel(new_df, 'Gender')


y_probas_new = pipeline.predict_proba(new_df)
threshold = 0.23
y_pred_new = (y_probas_new[:, 1] >= threshold).astype(int)


df_resultado['churn_predito'] = y_pred_new
df_resultado['churn_proba'] = y_probas_new[:,1]


df_resultado = df_resultado.loc[:, ['RowNumber', 'churn_predito', 'churn_proba']]

df_resultado.to_csv("previsoes_com_probabilidades.csv", index=False)