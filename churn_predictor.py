import pandas as pd
import joblib
from tratamento import *

class ChurnPredictor:
    def __init__(self, model_path, output_path, threshold=0.23):
        self.model_path = model_path
        self.output_path = output_path
        self.threshold = threshold
        self.pipeline = joblib.load(self.model_path)
    
    def preprocess_data(self, df, sep=';'):
        """
        Função para fazer o pré-processamento dos dados
        """
        # Copiar o DataFrame para evitar alterações no original
        df_resultado = df.copy()
        
        # Remover variáveis desnecessárias
        remover_variavel(df_resultado, 'RowNumber')
        remover_variavel(df_resultado, 'CustomerId')
        remover_variavel(df_resultado, 'Surname')
        remover_variavel(df_resultado, 'Gender')
        
        return df_resultado
    
    def make_predictions(self, df):
        """
        Função para gerar as previsões
        """
        # Gerar probabilidades de abandono (churn) e fazer previsões
        y_probas = self.pipeline.predict_proba(df)
        y_pred = (y_probas[:, 1] >= self.threshold).astype(int)
        
        return y_pred, y_probas
    
    def save_predictions(self, df, y_pred, y_probas):
        """
        Função para salvar os resultados das previsões em um arquivo CSV
        """
        df['predictedValues'] = y_pred
        df['churn_proba'] = y_probas[:, 1]
        
        # Garantir que não existam valores nulos
        df = df.fillna(0)
        
        # Selecionar as colunas desejadas para salvar no arquivo
        df_resultado = df.loc[:, ['RowNumber', 'predictedValues']]
        df_resultado['predictedValues'] = df_resultado['predictedValues'].fillna(0)

        # Salvar o arquivo
        df_resultado.to_csv(self.output_path, index=False)
        print(f"Previsões salvas em: {self.output_path}")
    
    def run(self, input_file, sep=';'):
        """
        Função principal para rodar todo o fluxo de predição
        """
        # Carregar o arquivo CSV
        df = pd.read_csv(input_file, sep=sep)
        
        # Pré-processar os dados
        df_preprocessed = self.preprocess_data(df, sep)
        
        # Fazer as previsões
        y_pred, y_probas = self.make_predictions(df_preprocessed)
        
        # Salvar os resultados
        self.save_predictions(df, y_pred, y_probas)

