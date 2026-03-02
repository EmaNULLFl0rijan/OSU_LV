""" Napišite program koji od korisnika zahtijeva unos brojeva u beskonacnoj petlji ˇ
sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
(npr. slovo umjesto brojke) na nacin da program zanemari taj unos i ispiše odgovaraju ˇ cu poruku. """

list = []
while True:
    x = input("Unesite broj (ili 'Done' za kraj): ")

    if x.lower() == "done":
        break
    elif x.isdigit():
        list.append(int(x))
    else:
        print("Neispravan unos, molimo unesite broj ili 'Done' za kraj.")

print("Brojevi koje ste unijeli su:", list)

print("Broj unesenih brojeva:", len(list))
print("Srednja vrijednost:", sum(list) / len(list))
print("Minimalna vrijednost:", min(list))
print("Maksimalna vrijednost:", max(list))
print("Sortirani brojevi su:", sorted(list))




