import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

data.plot.scatter(
    x='Fuel Consumption City (L/100km)',
    y='Fuel Consumption Hwy (L/100km)',
    c='Engine Size (L)',
    cmap='hot',
    s=50
)

plt.show()