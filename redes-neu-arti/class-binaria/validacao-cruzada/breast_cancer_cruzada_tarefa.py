# chegar, no mínimo, em 90% de precisão

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
# para a validação cruzada
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# carregamento da base de dados
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

# criando a validação cruzada
def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 30, activation = 'relu', 
                            kernel_initializer = 'random_uniform', input_dim = 30))
    # 'dropout' - zera neurônios da entrada, de uma forma aleatória
    # ajuda a prevenir o overfitting
    # entre 20 e 30% é o recomendado. Mais que isso, pode-se entrar no underfitting
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 16, activation = 'relu', 
                            kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 16, activation = 'relu', 
                            kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 8, activation = 'relu', 
                            kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    otimizador = keras.optimizers.adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', 
                          metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede, 
                                epochs = 100, batch_size = 9)

# realizando os testes várias vezes
# 'cv' (cross validation) - número de divisões da base de dados
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')

# acerto da base de dados
media = resultados.mean()

# verificando a variação dos resultados
# quanto maior o resultado, maior a tendência de ter overfitting
# se adaptando demais à base de dados
desvio = resultados.std()

