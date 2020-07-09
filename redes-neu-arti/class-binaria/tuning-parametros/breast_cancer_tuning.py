import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
# pesquisa em grade para descobrir os melhores parâmetros
from sklearn.model_selection import GridSearchCV

# carregamento da base de dados
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

# passando os parâmetros para modificação
def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer, input_dim = 30))
    # 'dropout' - zera neurônios da entrada, de uma forma aleatória
    # ajuda a prevenir o overfitting
    # entre 20 e 30% é o recomendado. Mais que isso, pode-se entrar no underfitting
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    # não há alteração na saída, nesse caso
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer = optimizer, loss = loss, 
                          metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
# parâmetros em forma de dicionário
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              # 'sgd' - descida do gradiente estocástico
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              # 'normal' - distribuição normal
              'kernel_initializer': ['random_uniforme', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(previsores, classe)
# com base nesses resultados, a versão "final" da rede é desenvolvida
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_