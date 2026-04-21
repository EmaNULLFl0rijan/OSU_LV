""" Skripta Zadatak_1.py ucitava CIFAR-10 skup podataka. Ovaj skup sadrži
50000 slika u skupu za ucenje i 10000 slika za testiranje. Slike su RGB i rezolucije su 32x32.
Svakoj slici je pridružena jedna od 10 klasa ovisno koji je objekt prikazan na slici. Potrebno je:
1. Proucite dostupni kod. Od kojih se slojeva sastoji CNN mreža? Koliko ima parametara
mreža?
2. Pokrenite ucenje mreže. Pratite proces ucenja pomocu alata Tensorboard na sljedeci nacin.
Pokrenite Tensorboard u terminalu pomocu naredbe:
tensorboard–logdir=logs
i zatim otvorite adresu http://localhost:6006/ pomocu web preglednika.
3. Proucite krivulje koje prikazuju tocnost klasifikacije i prosjecnu vrijednost funkcije gubitka
na skupu podataka za ucenje i skupu podataka za validaciju. Što se dogodilo tijekom ucenja
mreže? Zapišite tocnost koju ste postigli na skupu podataka za testiranje """

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn',
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





