""" Napišite Python skriptu koja ce ucitati tekstualnu datoteku naziva song.txt.
Potrebno je napraviti rjecnik koji kao kljuceve koristi sve razlicite rijeci koje se pojavljuju u datoteci, 
dok su vrijednosti jednake broju puta koliko se svaka rijec (kljuc) pojavljuje u datoteci. ˇ
Koliko je rijeci koje se pojavljuju samo jednom u datoteci? Ispišite ih. """

with open("song.txt") as file:
    content = file.read().lower()
    words = content.split()
    wordCount = {}
    for word in words:
        word = word.strip(",")
        wordCount[word] = wordCount.get(word, 0) + 1

    singleOccurrence = [word for word, count in wordCount.items() if count == 1]
    print("Rijeci koje se pojavljuju samo jednom:\n", singleOccurrence)
    print("\nBroj rijeci:", len(singleOccurrence))