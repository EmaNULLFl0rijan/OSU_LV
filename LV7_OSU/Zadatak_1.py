""" Skripta zadatak_1.py sadrži funkciju generate_data koja služi za generiranje
umjetnih podatkovnih primjera kako bi se demonstriralo grupiranje. Funkcija prima cijeli broj
koji definira željeni broju uzoraka u skupu i cijeli broj od 1 do 5 koji definira na koji nacin ce
se generirati podaci, a vraca generirani skup podataka u obliku numpy polja pri cemu su prvi i
drugi stupac vrijednosti prve odnosno druge ulazne velicine za svaki podatak. Skripta generira ˇ
500 podatkovnih primjera i prikazuje ih u obliku dijagrama raspršenja.
1. Pokrenite skriptu. Prepoznajete li koliko ima grupa u generiranim podacima? Mijenjajte
nacin generiranja podataka. 
2. Primijenite metodu K srednjih vrijednosti te ponovo prikažite primjere, ali svaki primjer
obojite ovisno o njegovoj pripadnosti pojedinoj grupi. Nekoliko puta pokrenite programski
kod. Mijenjate broj K. Što primjecujete? 
3. Mijenjajte nacin definiranja umjetnih primjera te promatrajte rezultate grupiranja podataka
(koristite optimalni broj grupa). Kako komentirate dobivene rezultate? """

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans

def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe (izdužene)
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe (različitih gustoća)
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe (koncentrične kružnice)
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe (polumjeseci)
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# Prolazimo kroz svih 5 načina generiranja podataka
for flagc in range(1, 6):
    X = generate_data(500, flagc)
    
    # Određivanje optimalnog broja grupa na temelju definicije u generate_data
    if flagc in [1, 2]:
        k = 3
    elif flagc == 3:
        k = 4
    else: # flagc 4 i 5
        k = 2

    # 2. Inicijalizacija i primjena algoritma K srednjih vrijednosti
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
    km.fit(X)
    labels = km.predict(X)
    
    # Prikaz primjera s bojama prema dodijeljenim klasterima
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='brg', edgecolors='k')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'K-means grupiranje (flagc = {flagc}, K = {k})')
    plt.show()