import pandas as pd
# separar a base de dados entre treino e validação
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
# camadas densas/ocultas
from keras.layers import Dense
# medir a taxa de acerto
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('input.csv')
classe = pd.read_csv('output.csv')

label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)

# test_size indica a porcentagem para a validação '25%' (75% para o treinamento)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)

classificador = Sequential()
# primeira camada oculta e de entrada
# units - quant. de neurônios (entradas + saidas / 2) -> 6 + 1 / 2 = 16
classificador.add(Dense(units = 4, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 6))
classificador.add(Dense(units = 4, activation = 'relu',
                        kernel_initializer = 'random_uniform'))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.summary()

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',                       metrics = ['binary_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 30, epochs = 1000)


previsoes = classificador.predict(previsores_teste)
# retornando valores bool 'True' ou 'False'
previsoes = (previsoes > 0.5)

# medindo a taxa de acerto - 'sklearn'
precisao = accuracy_score(classe_teste, previsoes)
# criando a matriz de confusão - linha (valor da classe) e coluna (valor da previsão)
matriz = confusion_matrix(classe_teste, previsoes)

# '0' indica o erro e '1' a precisão
resultado = classificador.evaluate(previsores_teste, classe_teste)