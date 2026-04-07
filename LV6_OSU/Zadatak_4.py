""" Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra C i γ
algoritma SVM za problem iz Zadatka 1. """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)
        
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

# --- ZADATAK 6.5.3: SVM model ---

# 1. Inicijalizacija SVM modela s RBF kernelom
# Početne vrijednosti: C=1.0, gamma='scale' (ili proizvoljno npr. 1.0)
svm_model = svm.SVC(kernel='rbf', C=1.0, gamma=1.0)
svm_model.fit(X_train_n, y_train)

# Evaluacija na testnom skupu
y_test_p_svm = svm_model.predict(X_test_n)
print(f"\nSVM (RBF, C=1, gamma=1) Točnost test: {accuracy_score(y_test, y_test_p_svm):.3f}")

# Prikaz granice odluke
plot_decision_regions(X_train_n, y_train, classifier=svm_model)
plt.title('SVM RBF kernel (C=1, gamma=1)')
plt.xlabel('Age (scaled)')
plt.ylabel('EstimatedSalary (scaled)')
plt.show()

# 2. EKSPERIMENT: Promjena hiperparametara
# Primjer A: Veliki C i veliki gamma (opasnost od overfittinga)
svm_overfit = svm.SVC(kernel='rbf', C=10.0, gamma=10.0)
svm_overfit.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=svm_overfit)
plt.title('SVM RBF (C=10, gamma=10) - Overfitting')
plt.show()

# Primjer B: Mali C i mali gamma (opasnost od underfittinga)
svm_underfit = svm.SVC(kernel='rbf', C=0.1, gamma=0.1)
svm_underfit.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=svm_underfit)
plt.title('SVM RBF (C=0.1, gamma=0.1) - Underfitting')
plt.show()

# 3. EKSPERIMENT: Promjena tipa kernela
# Linearni kernel
svm_linear = svm.SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_n, y_train)
plot_decision_regions(X_train_n, y_train, classifier=svm_linear)
plt.title('SVM Linearni kernel')
plt.show()