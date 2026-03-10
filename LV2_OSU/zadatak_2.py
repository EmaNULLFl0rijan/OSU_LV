""" Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
ženama. Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja data pri cemu je u ˇ
prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treci ´
stupac polja je masa u kg.
a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja? ˇ
b) Prikažite odnos visine i mase osobe pomocu naredbe ´ matplotlib.pyplot.scatter.
c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom ˇ
podatkovnom skupu.
e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili
muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka.
ind = (data[:,0] == 1) """

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data.csv", delimiter=",")

# a)
num_rows, num_cols = data.shape
print("Broj osoba: ", num_rows - 1)

# b)
plt.scatter(data[1:, 1], data[1:, 2])
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase")
plt.show()

# c)
plt.scatter(data[1::50, 1], data[1::50, 2])
plt.xlabel("Visina (cm)")   
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase (svaka pedeseta osoba)")
plt.show()

# d)
minheight = np.min(data[1:, 1])
maxheight = np.max(data[1:, 1])
meanheight = np.mean(data[1:, 1])
print("Minimalna visina: ", minheight)
print("Maksimalna visina: ", maxheight)
print("Srednja visina: ", meanheight)

# e)
ind = (data[:,0] == 1)
print("Muškarci:")
print("Minimalna visina: ", np.min(data[ind, 1]))
print("Maksimalna visina: ", np.max(data[ind, 1]))
print("Srednja visina: ", np.mean(data[ind, 1]))

ind = (data[:,0] == 0)
print("Žene:")
print("Minimalna visina: ", np.min(data[ind, 1]))
print("Maksimalna visina: ", np.max(data[ind, 1]))
print("Srednja visina: ", np.mean(data[ind, 1]))