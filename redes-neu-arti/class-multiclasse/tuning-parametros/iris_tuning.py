import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)

def criarRede(optimizer, kernel_initializer, activation, neurons, dropout):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer, input_dim = 4))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer))
    classificador.add(Dropout(dropout))
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', 
                          metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)

parametros = {'batch_size': [10, 20],
              'epochs': [1000, 1200],
              'optimizer': ['adam', 'sgd'],
              'kernel_initializer': ['random_uniforme', 'normal'],
              'activation': ['relu', 'tanh'],
              'dropout': [0.2, 0.3],
              'neurons': [4, 8]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           cv = 5)

grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_