import pandas as pd
import numpy as np
from tratamento import mudar_tipo_dados, remover_variavel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df = pd.read_csv("Abandono_clientes.csv", sep=',')

#remover as variáveis insignificantes para o modelo (já analisado no EDA)
remover_variavel(df, 'RowNumber')
remover_variavel(df, 'CustomerId')
remover_variavel(df, 'Surname')
remover_variavel(df, 'Gender') #Gender será removido para não enviezar ou deixar o modelo discriminante


continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

categorical_vars = [ 'Geography']




encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_vars)  # Drop para evitar dummy trap
    ],
    remainder='passthrough'
)

X_categorical_encoded = encoder.fit_transform(df[categorical_vars + ['HasCrCard', 'IsActiveMember']])


scaler = StandardScaler()
X_continuous_scaled = scaler.fit_transform(df[continuous_vars])

X_combined = np.hstack((X_continuous_scaled, X_categorical_encoded))


y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


