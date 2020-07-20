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
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 4, activation = 'tanh', 
                        kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', 
                          metrics = ['accuracy'])

classificador.fit(previsores, classe, batch_size = 20, epochs = 1000)

classificador_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)
    
classificador.save_weights('classificador_iris.h5')