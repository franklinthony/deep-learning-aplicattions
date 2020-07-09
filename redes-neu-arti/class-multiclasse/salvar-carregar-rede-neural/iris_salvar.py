import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)

classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'tanh', 
                        kernel_initializer = 'normal', input_dim = 4))
# 'dropout' - zera neurônios da entrada, de uma forma aleatória
# ajuda a prevenir o overfitting
# entre 20 e 30% é o recomendado. Mais que isso, pode-se entrar no underfitting
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 4, activation = 'tanh', 
                        kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', 
                          metrics = ['accuracy'])

# treinamento
classificador.fit(previsores, classe, batch_size = 20, epochs = 1000)

# salvando em disco
classificador_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)
    
# salvando os pesos da rede neural
classificador.save_weights('classificador_iris.h5')