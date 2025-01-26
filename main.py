from churn_predictor import ChurnPredictor

input_file = 'datasets/Abandono_teste.csv'
output_file = 'previsoes_com_probabilidades.csv'
model_file = 'modelo_treinado.pkl'

predictor = ChurnPredictor(model_path=model_file, output_path=output_file)

predictor.run(input_file, sep=';')
