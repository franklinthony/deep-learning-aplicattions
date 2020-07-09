import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

# parâmetros com os melhores resultados (tuning)
classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal', input_dim = 30))
# 'dropout' - zera neurônios da entrada, de uma forma aleatória
# ajuda a prevenir o overfitting
# entre 20 e 30% é o recomendado. Mais que isso, pode-se entrar no underfitting
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))
# não há alteração na saída, nesse caso
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                      metrics = ['binary_accuracy'])

# treinamento
classificador.fit(previsores, classe, batch_size = 10, epochs = 100)

# novo registro para análise (em forma de linha)
novo = np.array([[15.80, 8.34, 118, 900, 0.10,
                  0.26, 0.08, 0.134, 0.178, 0.20,
                  0.05, 1098, 0.87, 4500, 145.2,
                  0.005, 0.04, 0.05, 0.015, 0.03,
                  0.087, 23.15, 16.64, 178.5, 2018,
                  0.14, 0.185, 0.84, 158, 0.363]])

previsao = classificador.predict(novo)
previsao= (previsao > 0.9)