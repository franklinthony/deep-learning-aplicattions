import numpy as np
import pandas as pd
# fazendo o carregamento
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

arquivo = open('classificador_iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

# definindo o classificador
classificador = model_from_json(estrutura_rede)
# carregando os pesos
classificador.load_weights('classificador_iris.h5')

# novo registro para análise (em forma de linha)
novo = np.array([[3.0, 2.8, 4.6, 2.7]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.8)

# avaliando com base de testes
base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)

classificador.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', 
                          metrics = ['accuracy'])

resultado = classificador.evaluate(previsores, classe)
