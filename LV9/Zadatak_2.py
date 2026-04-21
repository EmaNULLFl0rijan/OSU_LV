""" Modificirajte skriptu iz prethodnog zadatka na nacin da na odgovarajuca mjesta u
mrežu dodate dropout slojeve. Prije pokretanja ucenja promijenite Tensorboard funkciju povratnog
poziva na nacin da informacije zapisuje u novi direktorij (npr. log_dir = '/log/cnn_dropout'). Pratite tijek
ucenja. Kako komentirate utjecaj dropout slojeva na performanse mreže? """

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Pripremi podatke (skaliraj ih na raspon [0,1])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train, dtype ="uint8")
y_test = to_categorical(y_test, dtype ="uint8")

# CNN mreza s dodanim Dropout slojem
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))

# Konvolucijski blokovi ostaju isti
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

# Potpuno povezani slojevi s Dropout-om
model.add(layers.Dense(500, activation='relu'))
# Dodan Dropout sloj koji iskljucuje 30% neurona tijekom ucenja
model.add(layers.Dropout(0.3)) 
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# Definiraj listu s funkcijama povratnog poziva s novim direktorijem
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn_dropout',
                                update_freq = 100)
]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_n,
          y_train,
          epochs = 40,
          batch_size = 64,
          callbacks = my_callbacks,
          validation_split = 0.1)

score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')