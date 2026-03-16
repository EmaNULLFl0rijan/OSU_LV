""" Napišite programski kod koji ce prikazati sljede ´ ce vizualizacije: ´
a) Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz. ´
b) Pomocu dijagrama raspršenja prikažite odnos izme ´ du gradske potrošnje goriva i emisije ¯
C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu ¯
velicina, obojite to ˇ ckice na dijagramu raspršenja s obzirom na tip goriva. ˇ
c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip ´
goriva. Primjecujete li grubu mjernu pogrešku u podacima? ´
d) Pomocu stup ´ castog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu ˇ
groupby.
e) Pomocu stup ´ castog grafa prikažite na istoj slici prosje ˇ cnu C02 emisiju vozila s obzirom na ˇ
broj cilindara. """

import pandas as pd
import matplotlib.pyplot as plt

# Učitavanje podataka
data = pd.read_csv("data_C02_emission.csv")

# Osnovno čišćenje podataka
data = data.dropna().drop_duplicates().reset_index(drop=True)

# Pretvorba kategorijskih varijabli u category
kategoricke = ["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type"]
for col in kategoricke:
    data[col] = data[col].astype("category")

# Mapa oznaka tipa goriva
fuel_labels = {
    "X": "Regular gasoline",
    "Z": "Premium gasoline",
    "D": "Diesel",
    "E": "Ethanol (E85)",
    "N": "Natural gas"
}

# a) Histogram emisije CO2
plt.figure(figsize=(8, 5))
data["CO2 Emissions (g/km)"].plot(kind="hist", bins=30, edgecolor="black")
plt.title("Histogram emisije CO2")
plt.xlabel("CO2 Emissions (g/km)")
plt.ylabel("Broj vozila")
plt.grid(axis="y", alpha=0.3)

# b) Dijagram raspršenja: gradska potrošnja vs CO2 emisija, obojano po tipu goriva
plt.figure(figsize=(8, 6))
for fuel, grupa in data.groupby("Fuel Type"):
    plt.scatter(
        grupa["Fuel Consumption City (L/100km)"],
        grupa["CO2 Emissions (g/km)"],
        s=30,
        alpha=0.7,
        label=f"{fuel} - {fuel_labels.get(fuel, fuel)}"
    )

plt.title("Gradska potrošnja goriva i emisija CO2")
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend(title="Tip goriva")
plt.grid(alpha=0.3)

# c) Kutijasti dijagram izvangradske potrošnje s obzirom na tip goriva
plt.figure(figsize=(8, 6))
data.boxplot(column="Fuel Consumption Hwy (L/100km)", by="Fuel Type")
plt.title("Izvangradska potrošnja po tipu goriva")
plt.suptitle("")  # miče automatski dodatni naslov
plt.xlabel("Tip goriva")
plt.ylabel("Fuel Consumption Hwy (L/100km)")
plt.grid(alpha=0.3)

# d) Stupčasti dijagram: broj vozila po tipu goriva (groupby)
plt.figure(figsize=(8, 5))
broj_vozila_po_gorivu = data.groupby("Fuel Type").size()
broj_vozila_po_gorivu.index = [fuel_labels.get(x, x) for x in broj_vozila_po_gorivu.index]
broj_vozila_po_gorivu.plot(kind="bar", edgecolor="black")
plt.title("Broj vozila po tipu goriva")
plt.xlabel("Tip goriva")
plt.ylabel("Broj vozila")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)

# e) Stupčasti graf: prosječna CO2 emisija s obzirom na broj cilindara
plt.figure(figsize=(8, 5))
prosjek_co2_po_cilindrima = data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()
prosjek_co2_po_cilindrima.plot(kind="bar", edgecolor="black")
plt.title("Prosječna CO2 emisija po broju cilindara")
plt.xlabel("Broj cilindara")
plt.ylabel("Prosječna CO2 emisija (g/km)")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()