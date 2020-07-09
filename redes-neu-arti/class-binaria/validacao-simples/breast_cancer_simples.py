import pandas as pd
# separar a base de dados entre treino e validação
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
# camadas densas/ocultas
from keras.layers import Dense
# medir a taxa de acerto
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

# test_size indica a porcentagem para a validação '25%' (75% para o treinamento)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)

# criação do classificador
classificador = Sequential()
# primeira camada oculta e de entrada
# units - quant. de neurônios (entradas + saidas / 2) -> 30 + 1 / 2 = 16
# kernel_initializer - inicialização dos pesos
# input_dim - quantidade de elementos na entrada
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))
# nova camada oculta
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform'))
# camada de saida
# sigmoid - retorna um valor entre 0 e 1 - classificação binária
classificador.add(Dense(units = 1, activation = 'sigmoid'))

# alterando parâmetros do otimizador
# lr - learning rate
# decay - valor de decremento
# clipvalue - congela o valor de decaimento entre -0.5 e 0.5
# alterá-los conforme os resultados
otimizador = keras.optimizers.adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', 
                      metrics = ['binary_accuracy'])

# compilação da rede neural
# optimizer - função para ajuste dos pesos (descida do gradiente estocástico, por exemplo) - 'adam' é o mais indicado
# loss - função de perda (cálculo de erro) - 'binary_crossentropy' mais utilizado em classificação binária
# metrics - métrica para a avaliação
# classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                      # metrics = ['binary_accuracy'])

# treinamento
# batch size - números de registros para calcular o erro e ajustar os pesos - '10' em '10', nesse caso
# os resultados podem ser diferentes, uma vez que a inicialização dos pesos é aleatória
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs = 50)

# visualização dos pesos
# '30' entradas, '16' neurônios na camada oculta
# 'bias' existente por default '(16, )' - 1 neurônio bias conectado aos 16 da camada oculta
pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))

# '16' na camada oculta ligados aos '16' na segunda camada oculta
# mais uma unidade de bias
pesos1 = classificador.layers[1].get_weights()

# '16' na última camada oculta ligados à camada de saída
# mais uma unidade de bias ligado à camada de saída
pesos2 = classificador.layers[2].get_weights()

# teste retornando valores de probabilidade
previsoes = classificador.predict(previsores_teste)
# retornando valores bool 'True' ou 'False'
previsoes = (previsoes > 0.5)

# medindo a taxa de acerto - 'sklearn'
precisao = accuracy_score(classe_teste, previsoes)
# criando a matriz de confusão - linha (valor da classe) e coluna (valor da previsão)
matriz = confusion_matrix(classe_teste, previsoes)

# medindo as taxas de acerto e de erro 'keras'
# '0' indica o erro e '1' a precisão
resultado = classificador.evaluate(previsores_teste, classe_teste)