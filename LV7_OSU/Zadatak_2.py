""" Kvantizacija boje je proces smanjivanja broja razlicitih boja u digitalnoj slici, ali
uzimajuci u obzir da rezultantna slika vizualno bude što slicnija originalnoj slici. Jednostavan
nacin kvantizacije boje može se postici primjenom algoritma K srednjih vrijednosti na RGB
vrijednosti elemenata originalne slike. Kvantizacija se tada postiže zamjenom vrijednosti svakog
elementa originalne slike s njemu najbližim centrom. Na slici 7.3a dan je primjer originalne
slike koja sadrži ukupno 106,276 boja, dok je na slici 7.3b prikazana rezultantna slika nakon
kvantizacije i koja sadrži samo 5 boja koje su odredene algoritmom K srednjih vrijednosti.
1. Otvorite skriptu zadatak_2.py. Ova skripta ucitava originalnu RGB slikutest_1.jpg
te ju transformira u podatkovni skup koji dimenzijama odgovara izrazu (7.2) pri cemu je n
broj elemenata slike, a m je jednak 3. Koliko je razlicitih boja prisutno u ovoj slici?
2. Primijenite algoritam K srednjih vrijednosti koji ce pronaci grupe u RGB vrijednostima
elemenata originalne slike.
3. Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadajucim centrom.
4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. Komentirajte dobivene
rezultate.
5. Primijenite postupak i na ostale dostupne slike.
6. Graficki prikažite ovisnosti o broju grupa K. Koristite atribut inertia objekta klase
KMeans. Možete li uociti lakat koji upucuje na optimalni broj grupa?
7. Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku. Što
primjecujete? """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# 1. Učitaj sliku
img = Image.imread("imgs\\test_6.jpg")

# Prikaz originalne slike
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# Pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# Transformiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w, h, d = img.shape
img_array = np.reshape(img, (w * h, d))

# --- TOČKA 1: Koliko je različitih boja prisutno u ovoj slici? ---
# np.unique pronalazi jedinstvene redove (boje) u 2D polju
broj_boja = len(np.unique(img_array, axis=0))
print(f"Broj različitih boja u originalnoj slici: {broj_boja}")

# --- TOČKA 2: Primjena algoritma K srednjih vrijednosti ---
# Odredi željeni broj grupa K (isprobaj mijenjati ovu vrijednost za Točku 4)
K = 5 
km = KMeans(n_clusters=K, init='k-means++', n_init=5, random_state=0)
km.fit(img_array)

# --- TOČKA 3: Zamjena vrijednosti svakog elementa pripadajućim centrom ---
# km.labels_ sadrži indeks grupe za svaki piksel
# km.cluster_centers_ sadrži RGB vrijednosti centara
img_array_aprox = km.cluster_centers_[km.labels_]

# --- TOČKA 4: Usporedba dobivene slike s originalnom ---
# Vraćanje aproksimiranog 2D polja natrag u 3D oblik slike
img_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title(f"Kvantizirana slika (K={K})")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

# --- TOČKA 6: Lakat metoda (ovisnost J o broju grupa K) ---
# Računamo inerciju (J) za različite vrijednosti K
# Napomena: Ovaj dio se može izvoditi malo duže
K_vrijednosti = range(1, 10)
J_vrijednosti = []

for k in K_vrijednosti:
    km_temp = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=0)
    km_temp.fit(img_array)
    J_vrijednosti.append(km_temp.inertia_) # inertia je vrijednost kriterijske funkcije J

plt.figure()
plt.plot(K_vrijednosti, J_vrijednosti, marker='o')
plt.xlabel('Broj grupa K')
plt.ylabel('Kriterijska funkcija J (Inertia)')
plt.title('Lakat metoda')
plt.show()

# --- TOČKA 7: Prikaz elemenata jedne grupe kao binarne slike ---
# Odabiremo jednu grupu (npr. grupu 0) i radimo masku
odabrana_grupa = 0
binarna_maska = (km.labels_ == odabrana_grupa)

# Reshape maske nazad u dimenzije (širina, visina)
binarna_slika = np.reshape(binarna_maska, (w, h))

plt.figure()
plt.title(f"Binarna slika za piksele u grupi {odabrana_grupa}")
plt.imshow(binarna_slika, cmap='gray')
plt.tight_layout()
plt.show()
