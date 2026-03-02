""" Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ SMSSpamCollection.txt
[1]. Ova datoteka sadrži 5574 SMS poruka pri cemu su neke ozna ˇ cene kao ˇ spam, a neke kao ham.
Primjer dijela datoteke:
ham Yup next stop.
ham Ok lar... Joking wif u oni...
spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!
a) Izracunajte koliki je prosje ˇ can broj rije ˇ ci u SMS porukama koje su tipa ham, a koliko je ˇ
prosjecan broj rije ˇ ci u porukama koje su tipa spam. ˇ
b) Koliko SMS poruka koje su tipa spam završava usklicnikom ? ˇ """

with open("SMSSpamCollection.txt") as f:
    content = f.read().lower()
    lines = content.splitlines()
    ham_words = []
    spam_words = []
    spam_exclamations = 0

    for line in lines:
        if line.startswith("ham"):
            ham_words.extend(line.split()[1:])
        elif line.startswith("spam"):
            spam_words.extend(line.split()[1:])
            if line.endswith("!"):
                spam_exclamations += 1

    avg_ham_words = len(ham_words) / len([line for line in lines if line.startswith("ham")])
    avg_spam_words = len(spam_words) / len([line for line in lines if line.startswith("spam")])

    print("Prosječan broj riječi u ham porukama:", avg_ham_words)
    print("Prosječan broj riječi u spam porukama:", avg_spam_words)
    print("Broj spam poruka koje završavaju usklicnikom:", spam_exclamations)