""" MNIST podatkovni skup za izgradnju klasifikatora rukom pisanih znamenki
dostupan je u okviru Keras-a. Skripta zadatak_1.py ucitava MNIST podatkovni skup te podatke
priprema za ucenje potpuno povezane mreže. 
1. Upoznajte se s ucitanim podacima. Koliko primjera sadrži skup za ucenje, a koliko skup za
testiranje? Kako su skalirani ulazni podaci tj. slike? Kako je kodirana izlazne velicina?
2. Pomocu matplotlib biblioteke prikažite jednu sliku iz skupa podataka za ucenje te ispišite
njezinu oznaku u terminal.
3. Pomocu klase Sequential izgradite mrežu prikazanu na slici 8.5. Pomocu metode
.summary ispišite informacije o mreži u terminal.
4. Pomocu metode .compile podesite proces treniranja mreže.
5. Pokrenite ucenje mreže (samostalno definirajte broj epoha i velicinu serije). Pratite tijek
ucenja u terminalu.
6. Izvršite evaluaciju mreže na testnom skupu podataka pomocu metode .evaluate.
7. Izracunajte predikciju mreže za skup podataka za testiranje. Pomocu scikit-learn biblioteke
prikažite matricu zabune za skup podataka za testiranje.
8. Pohranite model na tvrdi disk. """

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# prikazi nekoliko slika iz train skupa
# Prikazujemo prvu sliku iz skupa za učenje i ispisujemo njezinu oznaku
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Stvarna oznaka: {y_train[0]}")
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")

# pretvori labele (1-od-K kodiranje)
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape=input_shape))
# Sliku 28x28 pretvaramo u 1D vektor duljine 784 (potrebno za potpuno povezanu mrežu)
model.add(layers.Flatten()) 
# 1. skriveni sloj: 100 neurona, relu
model.add(layers.Dense(100, activation="relu"))
# 2. skriveni sloj: 50 neurona, relu
model.add(layers.Dense(50, activation="relu"))
# Izlazni sloj: 10 neurona (za 10 klasa), softmax
model.add(layers.Dense(num_classes, activation="softmax"))

print("\n--- Struktura modela ---")
model.summary()


# definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])


# provedi ucenje mreze
print("\n--- Započinjem učenje mreže ---")
batch_size = 32
epochs = 15

history = model.fit(x_train_s, 
                    y_train_s, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_split=0.1)


# Prikazi test accuracy i matricu zabune
print("\n--- Evaluacija modela ---")
score = model.evaluate(x_test_s, y_test_s, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Predikcija klasa za testni skup
y_pred = model.predict(x_test_s)
# Izlaz je u 1-od-K formatu, pa koristimo argmax da dobijemo konkretne klase (0-9)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_s, axis=1)

# Generiranje i ispis matrice zabune
cm = confusion_matrix(y_test_classes, y_pred_classes)
print("\nMatrica zabune:")
print(cm)


# spremi model
model.save("FCN_model.keras")
print("\nModel uspješno spremljen pod nazivom 'FCN_model.keras'.")