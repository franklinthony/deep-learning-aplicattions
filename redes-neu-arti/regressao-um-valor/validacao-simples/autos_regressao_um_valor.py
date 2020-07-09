import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense

# 'encoding' - refere-se ao ajuste dos atributos 'string' da base de dados
base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')
# apagando alguns atributos
# 'axis = 1' - indica a exclusão de toda a coluna
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

# analisando a coluna 'name'
base['name'].value_counts()
# como a coluna 'name' tem muitas variáveis, temos de apagar (evitar sobrecarga)
# não é importante, nesse contexto, por conta do desbalanceamento entre os dados
base = base.drop('name', axis = 1)
# igualmente para o 'seller' e 'offerType'
base['seller'].value_counts()
base = base.drop('seller', axis = 1)
base['offerType'].value_counts()
base = base.drop('offerType', axis = 1)

# tratando valores inconsistentes
# apagando ou alterando o preço, por exemplo
i1 = base.loc[base.price <= 10]
base.price.mean()
# excluindo...
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]

# tratando valores faltantes ('nan')
# apagando ou alterando pelo termo (veículo) que mais se repete - recurso para atributos categóricos
base.loc[pd.isnull(base['vehicleType'])]
# procurando qual veículo mais se repete
base['vehicleType'].value_counts() # limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() # manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() # golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() # benzin
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() # nein

# alterando os registros em branco pelos que mais aparecem
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

# usando 'label encoder' para transformar categorias em números
label_encoder_previsores = LabelEncoder()
# '[:, 0]' - todo a linha da coluna 'x'
previsores[:, 0] = label_encoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = label_encoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = label_encoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = label_encoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = label_encoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = label_encoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = label_encoder_previsores.fit_transform(previsores[:, 10])

# one hot encoder - variáveis tipo dummy
# 0 - 0 0 0
# 2 - 0 1 0
# 3 - 0 0 1
one_hot_encoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],
                                    remainder='passthrough') 
previsores = one_hot_encoder.fit_transform(previsores).toarray()

regressor = Sequential()
regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
regressor.add(Dense(units = 158, activation = 'relu'))
# 'linear' - problemas de regressao (default)
# vai retornar os valores calculados nas etapas anteriores
regressor.add(Dense(units = 1, activation = 'linear'))
regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam',
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)

previsoes = regressor.predict(previsores)
# media dos precos dos carros dessa base
preco_real.mean()
previsoes.mean()
