import pandas as pd

def mudar_tipo_dados(df, variavel, tipo):
    df[variavel] = df[variavel]. astype(tipo)
    



def remover_variavel(df, variavel):
    df.drop(variavel, axis=1, inplace=True)

