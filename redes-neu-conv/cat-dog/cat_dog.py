from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3),
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPool2D(pool_size = (2, 2)))
classificador.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3),
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPool2D(pool_size = (2, 2)))

classificador.add(Flatten())
 
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

# 'rescale' - normalizacao
gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')

base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'binary')

# se o poder de processamento for muito elevado
# pode-se colocar toda a base de dados - 4000, no caso
classificador.fit_generator(base_treinamento, steps_per_epoch = 4000 / 32,
                            epochs = 10, validation_data = base_teste,
                            validation_steps = 1000 / 32)

# treinamento com apenas uma imagem
imagem_teste = image.load_img('dataset/test_set/cachorro/dog.3500.jpg',
                              target_size = (64, 64))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
# numero de canais e quant. de imagens
imagem_teste = np.expand_dims(imagem_teste, axis = 0)
previsao = classificador.predict(imagem_teste)
# 'true' ou 'false'
previsao = (previsao > 0.5)
previsao = 'gato' if previsao == True else 'cachorro'
# quais as classes separadas?
base_treinamento.class_indices