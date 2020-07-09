from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
# pesquisa em grade para descobrir os melhores par�metros
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('input.csv')
classe = pd.read_csv('output.csv')

label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)

# passando os par�metros para modifica��o
def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer, input_dim = 6))
    # 'dropout' - zera neur�nios da entrada, de uma forma aleat�ria
    # ajuda a prevenir o overfitting
    # entre 20 e 30% � o recomendado. Mais que isso, pode-se entrar no underfitting
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    # n�o h� altera��o na sa�da, nesse caso
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer = optimizer, loss = loss, 
                          metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
# par�metros em forma de dicion�rio
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              # 'sgd' - descida do gradiente estoc�stico
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              # 'normal' - distribui��o normal
              'kernel_initializer': ['random_uniforme', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [4, 8]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(previsores, classe)
# com base nesses resultados, a vers�o "final" da rede � desenvolvida
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_