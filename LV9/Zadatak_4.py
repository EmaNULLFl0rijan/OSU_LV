""" Što se dogada s procesom ucenja:
1. ako se koristi jako velika ili jako mala velicina serije?
2. ako koristite jako malu ili jako veliku vrijednost stope ucenja?
3. ako izbacite odredene slojeve iz mreže kako biste dobili manju mrežu?
4. ako za 50% smanjite velicinu skupa za ucenje? """

""" 1. Jako velika veličina serije (batch size) može dovesti do bržeg učenja, ali može rezultirati lošijom generalizacijom modela. 
Jako mala veličina serije može dovesti do sporijeg učenja, ali može poboljšati generalizaciju modela. 
Optimalna veličina serije ovisi o problemu i modelu, ali često se koristi veličina serije između 32 i 256.

2. Jako mala vrijednost stope učenja može dovesti do sporog konvergiranja modela, dok 
jako velika vrijednost stope učenja može dovesti do divergiranja modela. Optimalna vrijednost stope učenja ovisi o problemu i 
modelu, ali često se koristi vrijednost između 0.001 i 0.01.

3. Izbacivanje određenih slojeva iz mreže može smanjiti kapacitet modela, 
što može dovesti do lošije performanse na složenim problemima. Međutim, može poboljšati performanse na 
jednostavnijim problemima i smanjiti rizik od prekomjernog učenja. Optimalna arhitektura mreže ovisi o problemu i modelu, 
ali često se koristi nekoliko konvolucijskih slojeva praćenih potpuno povezanim slojevima. 

4. Smanjenje veličine skupa za učenje može dovesti do lošije performanse modela,
jer model neće imati dovoljno podataka za učenje. Međutim, može smanjiti vrijeme učenja i rizik od prekomjernog učenja.
Optimalna veličina skupa za učenje ovisi o problemu i modelu, ali često se koristi što veći skup podataka za učenje.  """

