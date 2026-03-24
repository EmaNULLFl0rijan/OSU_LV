""" Skripta zadatak_1.py ucitava podatkovni skup iz data_C02_emission.csv.
Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih numerickih ulaznih velicina. Detalje oko ovog podatkovnog skupa mogu se prona ˇ ci u 3. ´
laboratorijskoj vježbi.
a) Odaberite željene numericke velicine specificiranjem liste s nazivima stupaca. Podijelite
podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%. 
b) Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova ´
o jednoj numerickoj velicini. Pri tome podatke koji pripadaju skupu za ucenje oznacite
plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom.
c) Izvršite standardizaciju ulaznih velicina skupa za ucenje. Prikažite histogram vrijednosti
jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja
transformirajte ulazne velicine skupa podataka za testiranje.
d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
povežite ih s izrazom 4.6.
e) Izvršite procjenu izlazne velicine na temelju ulaznih velicina skupa za testiranje. Prikažite
pomocu dijagrama raspršenja odnos izmedu stvarnih vrijednosti izlazne velicine i procjene
dobivene modelom.
f) Izvršite vrednovanje modela na nacin da izracunate vrijednosti regresijskih metrika na
skupu podataka za testiranje.
g) Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj
ulaznih velicina? """


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

#a)
data = pd.read_csv("data_C02_emission.csv")
numeric = ["Engine Size (L)",
           "Cylinders",
           "Fuel Consumption City (L/100km)",
           "Fuel Consumption Hwy (L/100km)",
           "Fuel Consumption Comb (L/100km)",
           "Fuel Consumption Comb (mpg)"]

X = data[numeric]
y = data["CO2 Emissions (g/km)"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.2,
                                                    random_state=1)

#b)
plt.scatter(X_train["Engine Size (L)"], y_train, color="blue")
plt.scatter(X_test["Engine Size (L)"], y_test, color="red")
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Engine Size vs CO2 Emissions (g/km)")
plt.show()

#c)
plt.hist(X_train["Engine Size (L)"], color="blue")
plt.show()

sc = StandardScaler()
#Converst data frame to numpy
X_train_scaled = sc.fit_transform(X_train)
#Numpy - Data frame
X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
#plots standardised values
plt.hist(X_train_scaled["Engine Size (L)"], color="blue")
plt.show()

X_test_scaled = sc.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)

#d)
linearModel = LinearRegression()
linearModel.fit(X_train_scaled, y_train)

#Pisitiv values have positive impact of y
#Negative values have negative impact on y
for col, coef in zip(X_train.columns, linearModel.coef_):
    print(col, coef)

#e)
y_test_prediction = linearModel.predict(X_test_scaled)
plt.scatter(y_test, y_test_prediction)
plt.plot([0,600], [0,600], color="red")
plt.title("Predicted CO2 Emissions (g/km)")
plt.xlabel("Real CO2 Emissions (g/km)")
plt.ylabel("Predicted CO2 Emissions (g/km)")
plt.show()

#f)
MSE = mean_squared_error(y_test, y_test_prediction)
RMSE = np.sqrt(MSE)
MAE = mean_absolute_error(y_test, y_test_prediction)
MAPE = mean_absolute_percentage_error(y_test, y_test_prediction)
R_2 = r2_score(y_test, y_test_prediction)

print("MSE: ", MSE)
print("RMSE: ", RMSE) #Mean error of model
print("MAE: ", MAE)
print("MAPE: ", MAPE)
print("R_2: ", R_2) #Precision of model