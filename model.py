import pandas as pd
import numpy as np
from tratamento import mudar_tipo_dados, remover_variavel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import joblib




df = pd.read_csv("datasets/Abandono_clientes.csv", sep=',')

remover_variavel(df, 'RowNumber')
remover_variavel(df, 'CustomerId')
remover_variavel(df, 'Surname')
remover_variavel(df, 'Gender') #Gender será removido para não enviezar ou deixar o modelo discriminante


continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

categorical_vars = [ 'Geography']

binary_vars = ['HasCrCard', 'IsActiveMember']


X = df.drop('Exited',axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



preprocessor = ColumnTransformer(
    transformers=[
        ('target_encoder', OneHotEncoder(), categorical_vars),
        ('scaler', StandardScaler(), continuous_vars),
        ('binary', 'passthrough', binary_vars)  
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(class_weight='balanced',
                                     random_state=42,
                                     n_estimators=100,
                                     min_samples_split=5,
                                     min_samples_leaf=2,
                                     max_depth=20,
                                     bootstrap=True))
])

pipeline.fit(X_train, y_train)


joblib.dump(pipeline, 'modelo_treinado.pkl')



