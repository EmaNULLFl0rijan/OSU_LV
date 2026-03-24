""" Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku 
varijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategorickih
velicina. Radi jednostavnosti nemojte skalirati ulazne velicine. Komentirajte dobivene rezultate.
Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
vozila radi? """

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, max_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np

data = pd.read_csv("data_C02_emission.csv")
input_variables = ['Fuel Type',
                   'Engine Size (L)',
                   'Cylinders',
                   'Fuel Consumption City (L/100km)',
                   'Fuel Consumption Hwy (L/100km)',
                   'Fuel Consumption Comb (L/100km)',
                   'Fuel Consumption Comb (mpg)'
                   ]
output = 'CO2 Emissions (g/km)'

print(data)

X = data[input_variables]
Y = data[output]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


ohe = OneHotEncoder()
#We get only fuel type in x_encoded_train
x_encoded_train = ohe.fit_transform(x_train[["Fuel Type"]]).toarray() #from data frame to numpy array
x_encoded_test = ohe.transform(x_test[["Fuel Type"]]).toarray()

print(x_encoded_train)
print(ohe.categories_) #Prints encoded types

#We only have a numeric data without fuel type
x_train_num = x_train.drop(["Fuel Type"], axis = 1)
x_test_num = x_test.drop(["Fuel Type"], axis = 1)

#Conected num data and fuel data one enx to the other
x_train_final = np.concatenate((x_train_num, x_encoded_train), axis = 1)
x_test_final = np.concatenate((x_test_num, x_encoded_test), axis = 1)
print(x_train_final.shape)

linear_regression = LinearRegression()
linear_regression.fit(x_train_final, y_train)

y_pred = linear_regression.predict(x_test_final)
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color = 'red')
plt.show()
#Evaluation of model
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
MAE = mean_absolute_error(y_test, y_pred)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
print("MSE: ", MSE)
print("RMSE: ", RMSE)
print("MAE: ", MAE)
print("MAPE: ", MAPE)
print("R2: ", R2)

#index of evry row is the same from original data red from csv
errors = np.abs(y_test - y_pred)
max_error_id = np.argmax(errors)
print(max_error_id)
print(x_test_final[max_error_id])
original_index = x_test.index[max_error_id]
print(data.loc[original_index])