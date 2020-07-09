import pandas as pd
# separar a base de dados entre treino e valida��o
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
# camadas densas/ocultas
from keras.layers import Dense
# medir a taxa de acerto
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

# test_size indica a porcentagem para a valida��o '25%' (75% para o treinamento)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)

# cria��o do classificador
classificador = Sequential()
# primeira camada oculta e de entrada
# units - quant. de neur�nios (entradas + saidas / 2) -> 30 + 1 / 2 = 16
# kernel_initializer - inicializa��o dos pesos
# input_dim - quantidade de elementos na entrada
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))
# nova camada oculta
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform'))
# camada de saida
# sigmoid - retorna um valor entre 0 e 1 - classifica��o bin�ria
classificador.add(Dense(units = 1, activation = 'sigmoid'))

# alterando par�metros do otimizador
# lr - learning rate
# decay - valor de decremento
# clipvalue - congela o valor de decaimento entre -0.5 e 0.5
# alter�-los conforme os resultados
otimizador = keras.optimizers.adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', 
                      metrics = ['binary_accuracy'])

# compila��o da rede neural
# optimizer - fun��o para ajuste dos pesos (descida do gradiente estoc�stico, por exemplo) - 'adam' � o mais indicado
# loss - fun��o de perda (c�lculo de erro) - 'binary_crossentropy' mais utilizado em classifica��o bin�ria
# metrics - m�trica para a avalia��o
# classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                      # metrics = ['binary_accuracy'])

# treinamento
# batch size - n�meros de registros para calcular o erro e ajustar os pesos - '10' em '10', nesse caso
# os resultados podem ser diferentes, uma vez que a inicializa��o dos pesos � aleat�ria
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs = 50)

# visualiza��o dos pesos
# '30' entradas, '16' neur�nios na camada oculta
# 'bias' existente por default '(16, )' - 1 neur�nio bias conectado aos 16 da camada oculta
pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))

# '16' na camada oculta ligados aos '16' na segunda camada oculta
# mais uma unidade de bias
pesos1 = classificador.layers[1].get_weights()

# '16' na �ltima camada oculta ligados � camada de sa�da
# mais uma unidade de bias ligado � camada de sa�da
pesos2 = classificador.layers[2].get_weights()

# teste retornando valores de probabilidade
previsoes = classificador.predict(previsores_teste)
# retornando valores bool 'True' ou 'False'
previsoes = (previsoes > 0.5)

# medindo a taxa de acerto - 'sklearn'
precisao = accuracy_score(classe_teste, previsoes)
# criando a matriz de confus�o - linha (valor da classe) e coluna (valor da previs�o)
matriz = confusion_matrix(classe_teste, previsoes)

# medindo as taxas de acerto e de erro 'keras'
# '0' indica o erro e '1' a precis�o
resultado = classificador.evaluate(previsores_teste, classe_teste)