import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

(X_treinamento, y_treinamento), (X_teste, y_teste) = cifar10.load_data()
plt.imshow(X_treinamento[5])
plt.title('Classe ' + str(y_treinamento[5]))
          
# 32 por 32 pixels, 3 canais (RGB)
# reduzindo a dimensionalidade
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               32, 32, 3)
previsores_teste = X_teste.reshape(X_teste.shape[0], 32, 32, 3)
# alterando o tipo de previsores para 'float'
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

# diminuindo a escala de cinza - normalizacao
previsores_treinamento /= 255
previsores_teste /= 255

# modificando as classes para tipo dummy (variáveis binárias)
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

# construindo a rede neural
classificador = Sequential()
# gerando a camada de convolucao
# '32' mapas de caracteristicas serao gerados
# '3, 3' - tamanho do detector de caracteristicas
classificador.add(Conv2D(64, (3, 3), input_shape = [32, 32, 3],
                         activation = 'relu'))
classificador.add(BatchNormalization())
# acrescetando o pooling
classificador.add(MaxPooling2D(pool_size = (2, 2)))

# mais uma camada de convol - nao precisa do 'input_shape'
classificador.add(Conv2D(64, (3, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2, 2)))
# flattening - matriz em vetor - so usa uma vez
classificador.add(Flatten())

# gerando a rede neural em si
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam', metrics = ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 128, epochs = 5,
                  # ja faz a validacao dos resultados
                  validation_data = [previsores_teste, classe_teste])

# equivale ao parametro 'validation_data'
#resultado = classificador.evaluate(previsores_teste, classe_teste)