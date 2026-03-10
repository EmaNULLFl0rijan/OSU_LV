""" Skripta zadatak_3.py ucitava sliku ’ ˇ road.jpg’. Manipulacijom odgovarajuce ´
numpy matrice pokušajte:
a) posvijetliti sliku,
b) prikazati samo drugu cetvrtinu slike po širini, ˇ
c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
d) zrcaliti sliku. """

import numpy as np
import matplotlib.pyplot as plt
img = plt.imread("road.jpg")
img = img[:,:,0].copy()

# a)
img_light = img * 1.3
img_light = np.clip(img_light, 0, 255)

plt.title("Posvijetljena slika")
plt.imshow(img_light, cmap="gray")
plt.show()

# b)
h, w = img.shape

second_quarter = img[:, w//4:w//2]

plt.imshow(second_quarter, cmap="gray")
plt.show()

# c)
img_rotated = np.rot90(img)
plt.title("Zarotirana slika")
plt.imshow(img_rotated, cmap="gray")
plt.show()

# d)
img_flipped = np.fliplr(img)
plt.title("Zrcaljena slika")
plt.imshow(img_flipped, cmap="gray")
plt.show()
