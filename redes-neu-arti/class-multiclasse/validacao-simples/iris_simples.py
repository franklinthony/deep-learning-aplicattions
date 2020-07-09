import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
# modificar os previsores categóricos por números
from sklearn.preprocessing import LabelEncoder
# modificando as saídas para um array de três elementos
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix

base = pd.read_csv('iris.csv')
# dividindo os previsores e as classes
# do 0 até o 3
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)
# iris setosa       1 0 0
# iris virginica    0 1 0
# iris versicolor   0 0 1

# dividindo entre treinamento e teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size = 0.25)

# criando a estrutura da rede neural
classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dense(units = 4, activation = 'relu'))
# 'softmax' para prob com mais de duas classes
# gera uma probabilidade para cada classe
classificador.add(Dense(units = 3, activation = 'softmax'))
# 'categorical_crossentropy' - prob com mais de duas classes
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs = 1000)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
# ajuste da matriz de confusão para multiclasses de saída
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]
matriz = confusion_matrix(previsoes2, classe_teste2)