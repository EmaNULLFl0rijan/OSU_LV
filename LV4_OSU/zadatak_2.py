""" Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku 
varijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategorickih
velicina. Radi jednostavnosti nemojte skalirati ulazne velicine. Komentirajte dobivene rezultate.
Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
vozila radi? """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

data = pd.read_csv("LV4/data_C02_emission.csv")

num_features = [
    "Engine Size (L)",
    "Cylinders",
    "Fuel Consumption City (L/100km)",
    "Fuel Consumption Hwy (L/100km)",
    "Fuel Consumption Comb (L/100km)",
    "Fuel Consumption Comb (mpg)"
]
cat_features = ["Fuel Type"]

x_num = data[num_features]
x_cat = data[cat_features]

ohe = OneHotEncoder()
x_cat_en = ohe.fit_transform(x_cat).toarray()

x = np.hstack([x_num.values, x_cat_en])
y = data["CO2 Emissions (g/km)"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

linearModel = lm.LinearRegression()
linearModel.fit(x_train, y_train)
print("Koeficijenti:",linearModel.coef_)
print("Slobodan član:",linearModel.intercept_)

y_predicted = linearModel.predict(x_test)

plt.figure()
plt.scatter(y_test, y_predicted, marker=".", color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="green")
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.show()

MAE = mean_absolute_error(y_test, y_predicted)
MSE = mean_squared_error(y_test, y_predicted)
RMSE = root_mean_squared_error(y_test, y_predicted)
R2 = r2_score(y_test, y_predicted)
print("MAE:", MAE)
print("MSE:", MSE)
print("RMSE:", RMSE)
print("R2:", R2)

errors = np.abs(y_test-y_predicted)
max_error = errors.max()
max_index = errors.idxmax()
print("Najveća pogreška:", max_error)
print(data.iloc[max_index])
