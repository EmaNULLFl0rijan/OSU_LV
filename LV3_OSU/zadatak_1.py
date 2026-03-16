"""Skripta zadatak_1.py ucitava podatkovni skup iz data_C02_emission.csv. 
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljedeca pitanja:´ 
a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? 
Postoje li izostale ili duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke velicine konvertirajte u tip category. 
b) Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? 
Ispišite u terminal: ´ ime proizvodaca, model vozila i kolika je gradska potrošnja.
 c) Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L? Kolika je prosjecna C02 emisija plinova za ova vozila? 
 d) Koliko mjerenja se odnosi na vozila proizvodaca Audi?
   Kolika je prosjecna emisija C02 plinova automobila proizvodaca Audi koji imaju 4 cilindara? ˇ 
e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na broj cilindara? 
f) Kolika je prosjecna gradska potrošnja u slu ˇ caju vozila koja koriste dizel, a kolika za vozila ˇ 
   koja koriste regularni benzin? Koliko iznose medijalne vrijednosti? g) Koje vozilo s 4 cilindra koje koristi 
   dizelski motor ima najvecu gradsku potrošnju goriva?
h) Koliko ima vozila ima rucni tip mjenjaca 
   (bez obzira na broj brzina)?
i) Izracunajte korelaciju izmedu numerickih velicina. Komentirajte dobiveni rezultat. """

import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

# a)
print("Broj mjerenja:", len(data))
print("Tipovi veličina prije pretvorbe:\n", data.dtypes)
print("Broj izostalih vrijednosti:\n", data.isnull().sum())
print("Broj dupliciranih vrijednosti:", data.duplicated().sum())

data = data.drop_duplicates()
data = data.dropna()

# kategoricke velicine u category
categorical_columns = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
for col in categorical_columns:
    data[col] = data[col].astype('category')

print("Tipovi veličina nakon pretvorbe:\n", data.dtypes)

# b)
largest_city = data.sort_values(by='Fuel Consumption City (L/100km)', ascending=False)
smallest_city = data.sort_values(by='Fuel Consumption City (L/100km)', ascending=True)

print("\nTri automobila s najvećom gradskom potrošnjom:")
print(largest_city[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3))

print("\nTri automobila s najmanjom gradskom potrošnjom:")
print(smallest_city[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3))

# c)
filteredData = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print("\nBroj vozila s motorom između 2.5 i 3.5 L:", len(filteredData))
print("Prosjek CO2 emisije:", filteredData['CO2 Emissions (g/km)'].mean())

# d)
audiData = data[data['Make'] == 'Audi']
print("\nBroj vozila proizvođača Audi:", len(audiData))

audi4CylinderData = audiData[audiData['Cylinders'] == 4]
print("Prosjek CO2 emisije za Audi s 4 cilindra:", audi4CylinderData['CO2 Emissions (g/km)'].mean())

# e)
cylinderCounts = data['Cylinders'].value_counts().sort_index()
print("\nBroj vozila po broju cilindara:\n", cylinderCounts)

cylinderEmissions = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print("\nProsječna emisija CO2 po broju cilindara:\n", cylinderEmissions)

# f)
dieselData = data[data['Fuel Type'] == 'D']
gasolineData = data[data['Fuel Type'] == 'X']

print("\nProsječna gradska potrošnja za dizel vozila:", dieselData['Fuel Consumption City (L/100km)'].mean())
print("Prosječna gradska potrošnja za vozila na regularni benzin:", gasolineData['Fuel Consumption City (L/100km)'].mean())

print("Medijan gradske potrošnje za dizel vozila:", dieselData['Fuel Consumption City (L/100km)'].median())
print("Medijan gradske potrošnje za vozila na regularni benzin:", gasolineData['Fuel Consumption City (L/100km)'].median())

# g)
diesel4CylinderData = data[(data['Fuel Type'] == 'D') & (data['Cylinders'] == 4)]
maxConsumptionVehicle = diesel4CylinderData.sort_values(by='Fuel Consumption City (L/100km)', ascending=False)

print("\nDizelsko vozilo s 4 cilindra i najvećom gradskom potrošnjom:")
print(maxConsumptionVehicle[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(1))

# h)
manualTransmissionData = data[data['Transmission'].astype(str).str.startswith('M')]
print("\nBroj vozila s ručnim mjenjačem:", len(manualTransmissionData))

# i)
numericalData = data.select_dtypes(include=['float64', 'int64'])
correlationMatrix = numericalData.corr()
print("\nKorelacija između numeričkih veličina:\n", correlationMatrix)

print("\nKomentar:")
print("Pozitivna korelacija bliska 1 znači da dvije veličine rastu zajedno.")
print("Negativna korelacija bliska -1 znači da jedna raste dok druga pada.")
print("Vrijednosti blizu 0 znače slabu ili nikakvu linearnu povezanost.")
print("Očekuje se jača pozitivna korelacija između veličine motora, broja cilindara, potrošnje goriva i CO2 emisije.")