#Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je placen ´
#po radnom satu. Koristite ugradenu Python metodu ¯ input(). Nakon toga izracunajte koliko ˇ
#je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na nacin da ukupni iznos ˇ
#izracunavate u zasebnoj funkciji naziva ˇ total_euro.

workHours = input("Radni sati: ")
hourlyRate = input("eura/h: ")

workHours = workHours.rstrip(" h ")

try:
    print(float(workHours) * float(hourlyRate), "eura")
except ValueError:
    print("Neispravan unos")