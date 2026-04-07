""" Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
te prikažite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ. Kako promjena
ovih hiperparametara utjece na granicu odluke te pogrešku na skupu podataka za testiranje?
Mijenjajte tip kernela koji se koristi. Što primjecujete? """

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

# 1. Inicijalni SVM s RBF kernelom (default vrijednosti ili C=1, gamma=1) [cite: 216]
SVM_model_rbf = svm.SVC(kernel='rbf', gamma=1, C=1)
SVM_model_rbf.fit(X_train_n, y_train)

y_test_p_SVM = SVM_model_rbf.predict(X_test_n)
print(f"\nSVM (RBF kernel, C=1, gamma=1) Tocnost: {accuracy_score(y_test, y_test_p_SVM):.3f}")

plot_decision_regions(X_train_n, y_train, classifier=SVM_model_rbf)
plt.title('SVM RBF kernel (C=1, gamma=1)')
plt.xlabel('x_1 (Dob)')
plt.ylabel('x_2 (Placa)')
plt.show()

# 2. Utjecaj promjene hiperparametara C i gamma
# Probajmo s vrlo velikim C i velikim gamma [cite: 109]
SVM_model_overfit = svm.SVC(kernel='rbf', gamma=10, C=10)
SVM_model_overfit.fit(X_train_n, y_train)

plot_decision_regions(X_train_n, y_train, classifier=SVM_model_overfit)
plt.title('SVM RBF kernel (C=10, gamma=10) - Overfitting')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()

# 3. Promjena tipa kernela (Linearni kernel) [cite: 290]
SVM_model_lin = svm.SVC(kernel='linear', C=1)
SVM_model_lin.fit(X_train_n, y_train)

plot_decision_regions(X_train_n, y_train, classifier=SVM_model_lin)
plt.title('SVM Linearni kernel')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()