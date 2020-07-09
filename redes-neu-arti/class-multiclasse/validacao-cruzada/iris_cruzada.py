import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
    classificador.add(Dense(units = 4, activation = 'relu'))
    # 'softmax' para prob com mais de duas classes
    # gera uma probabilidade para cada classe
    classificador.add(Dense(units = 3, activation = 'softmax'))
    # 'categorical_crossentropy' - prob com mais de duas classes
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede, epochs = 1000,
                                batch_size = 10)

resultados = cross_val_score(estimator = classificador, X = previsores,
                             y = classe, cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()