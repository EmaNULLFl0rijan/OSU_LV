
""" Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja
nekakvu ocjenu i nalazi se izmedu 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju ¯
sljedecih uvjeta: ´
>= 0.9 A
>= 0.8 B
>= 0.7 C
>= 0.6 D
< 0.6 F
Ako korisnik nije utipkao broj, ispišite na ekran poruku o grešci (koristite try i except naredbe).
Takoder, ako je broj izvan intervala [0.0 i 1.0] potrebno je ispisati odgovaraju ¯ cu poruku. """

x = input("Unesite ocjenu (izmedju 0.0 i 1.0): ")
try:
    x = float(x)
    if 0.0 <= x <= 1.0:
        print("Unesena ocjena je:", x)
        if x >= 0.9:
            print("Ocjena: A")
        elif x >= 0.8:
            print("Ocjena: B")
        elif x >= 0.7:
            print("Ocjena: C")
        elif x >= 0.6:
            print("Ocjena: D")
        else:
            print("Ocjena: F")
    else:
        print("Ocjena nije u rasponu od 0.0 do 1.0")
except ValueError:
    print("Neispravan unos")
        
               





