import numpy as np
import pandas as pd
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

arquivo = open('classificador_iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_iris.h5')

novo = np.array([[3.0, 2.8, 4.6, 2.7]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.8)

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)

classificador.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', 
                          metrics = ['accuracy'])

resultado = classificador.evaluate(previsores, classe)