""" Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra K
algoritma KNN za podatke iz Zadatka 1. """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

# --- ZADATAK 6.5.2: Unakrsna validacija za KNN ---
from sklearn.model_selection import cross_val_score

# Testiramo K od 1 do recimo 40
k_range = range(1, 40)
cv_scores = []

# Iteriramo kroz svaku vrijednost K i radimo 5-struku unakrsnu validaciju
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # cv=5 dijeli skup za učenje na 5 podskupova 
    scores = cross_val_score(knn, X_train_n, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Prikaz rezultata unakrsne validacije 
plt.figure()
plt.plot(k_range, cv_scores)
plt.xlabel('Broj susjeda K')
plt.ylabel('CV tocnost')
plt.title('Odredivanje optimalnog K (KNN)')
plt.grid(True)
plt.show()

# Pronalazak i ispis optimalnog K
optimal_k = k_range[np.argmax(cv_scores)]
print(f"Optimalni broj susjeda K prema unakrsnoj validaciji je: {optimal_k}")