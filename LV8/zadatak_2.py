""" Napišite skriptu koja ce u ´ citati izgra ˇ denu mrežu iz zadatka 1 i MNIST skup ¯
podataka. Pomocu matplotlib biblioteke potrebno je prikazati nekoliko loše klasi ´ ficiranih slika iz
skupa podataka za testiranje. Pri tome u naslov slike napišite stvarnu oznaku i oznaku predvidenu ¯
mrežom. """

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# 1. Učitavanje MNIST skupa podataka (samo testni dio je potreban za evaluaciju)
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Priprema (skaliranje) ulaznih podataka na isti način kao prilikom učenja
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

# 2. Učitavanje izgrađene mreže iz zadatka 1
# Napomena: Ako je model spremljen pod drugim imenom (npr. "FCN/"), prilagodite putanju.
model = keras.models.load_model("FCN_model.keras")

# 3. Predikcija klasa za cijeli testni skup
y_pred = model.predict(x_test_s)
# Izlaz mreže su vjerojatnosti za svaku klasu, uzimamo indeks najveće vjerojatnosti
y_pred_classes = np.argmax(y_pred, axis=1)

# 4. Pronalazak indeksa primjera koji su loše klasificirani
incorrect_indices = np.nonzero(y_pred_classes != y_test)[0]

print(f"Ukupno pogrešno klasificiranih primjera: {len(incorrect_indices)}")

# 5. Prikaz nekoliko loše klasificiranih slika
plt.figure(figsize=(12, 5))
num_images_to_show = 5

for i, incorrect_index in enumerate(incorrect_indices[:num_images_to_show]):
    plt.subplot(1, num_images_to_show, i + 1)
    # Koristimo originalni x_test za prikaz kako bi slike bile u rasponu 0-255
    plt.imshow(x_test[incorrect_index], cmap='gray')
    plt.title(f"Stvarno: {y_test[incorrect_index]}\nPredviđeno: {y_pred_classes[incorrect_index]}")
    plt.axis('off')

plt.tight_layout()
plt.show()