""" Napišite skriptu koja ce u ´ citati izgra ˇ denu mrežu iz zadatka 1. Nadalje, skripta ¯
treba ucitati sliku ˇ test.png sa diska. Dodajte u skriptu kod koji ce prilagoditi sliku za mrežu, ´
klasificirati sliku pomocu izgra ´ dene mreže te ispisati rezultat u terminal. Promijenite sliku ¯
pomocu nekog gra ´ fickog alata (npr. pomo ˇ cu Windows Paint-a nacrtajte broj 2) i ponovo pokrenite ´
skriptu. Komentirajte dobivene rezultate za razlicite napisane znamenke """

import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# 1. Učitavanje izgrađene mreže iz zadatka 1
# Prilagodite putanju ako ste model spremili pod drugim imenom
model = keras.models.load_model("FCN_model.keras")

# 2. Učitavanje slike s diska (npr. test.png)
# color_mode="grayscale" osigurava da slika ima samo jedan kanal
# target_size=(28, 28) automatski smanjuje sliku na potrebnu rezoluciju
image_path = "broj.png"

try:
    img = load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    
    # Pretvaranje slike u numpy polje
    img_array = img_to_array(img)
    
    # 3. Prilagodba slike za mrežu
    # VAŽNO: MNIST slike imaju crnu pozadinu (0) i bijela slova (255).
    # Ako ste crtali crnim kistom po bijeloj pozadini u Paintu, sliku morate invertirati!
    # Ako je vaša slika već crna s bijelim brojem, obrišite sljedeću liniju koda:
    img_array = 255.0 - img_array 
    
    # Skaliranje na raspon [0,1] kao u treningu
    img_array = img_array.astype("float32") / 255.0
    
    # Dodavanje "batch" dimenzije (mreža očekuje oblik (broj_slika, 28, 28, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prikaz slike koju šaljemo u mrežu (kako bismo bili sigurni da izgleda kao MNIST)
    plt.imshow(img_array[0, :, :, 0], cmap='gray')
    plt.title("Slika nakon predobrade")
    plt.axis('off')
    plt.show()

    # 4. Klasifikacija slike
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Ispis rezultata u terminal
    print("\n--- REZULTAT KLASIFIKACIJE ---")
    print(f"Mreža je prepoznala znamenku: {predicted_class}")
    print(f"Vjerojatnosti po klasama (0-9):\n{np.round(prediction[0], 3)}")

except FileNotFoundError:
    print(f"Pogreška: Slika '{image_path}' nije pronađena. Provjerite putanju i ime datoteke.")