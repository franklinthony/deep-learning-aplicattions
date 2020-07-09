import pandas as pd
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

# salvando em disco
classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)
    
# salvando os pesos da rede neural
classificador.save_weights('classificador_breast.h5')