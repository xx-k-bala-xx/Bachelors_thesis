import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from interaktywne_okno_kod_do_pracy import do_mask_on_image  
import matplotlib.patches as patches

# wczytywanie obrazów
image1 = cv2.imread("C:\\Users\\klaud\\Downloads\\zdj_do_tf\\monety10.jpg",
                    cv2.IMREAD_GRAYSCALE)
image1 = np.double(image1)

# ustalenie progow
p1, p2 = 100, 70

# redukcja szumu
gausian_kernel_3x3 = 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
il_g = 2
for i in range(il_g):
    image1 = do_mask_on_image(image1, gausian_kernel_3x3)

# maski Sobela
v_sobel_mask = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

h_sobel_mask = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
dx = do_mask_on_image(image1, v_sobel_mask)
dy = do_mask_on_image(image1, h_sobel_mask)

modul_of_gradient = np.zeros(dx.shape)
for i in range(modul_of_gradient.shape[0]):
    for j in range(modul_of_gradient.shape[1]):
        modul_of_gradient[i][j] = ((dx[i][j])**2 + (dy[i][j])**2)**(1/2)

# kierunek gradientu
angle2 = np.arctan2(dy, dx)
angle_of_gradient = angle2.copy()

# kwantyzacja kierunkow
for i in range(angle_of_gradient.shape[0]):
    for j in range(angle_of_gradient.shape[1]):
        if angle_of_gradient[i][j] < 0:
            angle_of_gradient[i][j] += np.pi
        if angle_of_gradient[i][j] <= np.pi/8 or angle_of_gradient[i][j] > 7*np.pi/8:
            angle_of_gradient[i][j] = 1  # E-W(poziomy)
        elif angle_of_gradient[i][j] <= 3*np.pi/8:
            angle_of_gradient[i][j] = 2
            # normalnie NE-SW, ale przy zmianie orientacji osi to NW-SE
        elif angle_of_gradient[i][j] <= 5*np.pi/8:
            angle_of_gradient[i][j] = 3  # N-S (pionowy)
        else:
            angle_of_gradient[i][j] = 4
            # normalnie NW-SE, ale przy orientacji osi OY w dół to NE-SW

# szukanie lokalnych maksimow
max_loc_grad = np.zeros(image1.shape)

for i in range(1, image1.shape[0]-1):
    for j in range(1, image1.shape[1]-1):
        if angle_of_gradient[i][j] == 1:
            u, v = 0, 1
        elif angle_of_gradient[i][j] == 2:
            u, v = 1, 1
        elif angle_of_gradient[i][j] == 3:
            u, v = 1, 0
        else:
            u, v = 1, -1
        if ((modul_of_gradient[i][j] >= modul_of_gradient[i+u][j+v]) and
                (modul_of_gradient[i][j] >= modul_of_gradient[i-u][j-v])):
            max_loc_grad[i][j] = modul_of_gradient[i][j]

# wyznaczenie slabych i silnych krawedzi
strong_edges = np.zeros(max_loc_grad.shape)
weak_edges = np.zeros(max_loc_grad.shape)
for i in range(0, image1.shape[0]-1):
    for j in range(0, image1.shape[1]-1):
        if max_loc_grad[i][j] >= p1:
            strong_edges[i][j] = 255
        elif max_loc_grad[i][j] >= p2:
            weak_edges[i][j] = 255
        else:
            pass

# szukanie krawedzi z histereza
for k in range(1, 4):
    for i in range(1, image1.shape[0] - 1):
        for j in range(1, image1.shape[1] - 1):
            if weak_edges[i][j] == 255:
                if_any_strong = False
                for u in range(-1, 2):
                    for v in range(-1, 2):
                        if strong_edges[i+u][j+v] == 255:
                            if_any_strong = True
                            break
                    if if_any_strong:
                        break
                strong_edges[i][j] = 255
                weak_edges[i][j] = 0
edges = strong_edges

# wykrywanie okregow, wersja z rysowaniem pkt z okręgu tylko w kierunku gradientu

# ustawianie parametrow
rmin = 120
rmax = 200
threshold = 0.2

acc_space = np.zeros(image1.shape)
acc = defaultdict(int)
for x in range(edges.shape[1]):
    for y in range(edges.shape[0]):
        if edges[y][x] != 0:
            teta = angle2[y][x]
            if teta < 0:
                teta += np.pi
            for r in range(rmin, rmax+1):
                for alpha in [teta, teta+np.pi]:
                    a = x - int(r*np.cos(alpha))
                    b = y - int(r*np.sin(alpha))
                    acc[(a, b, r)] += 1
                    if (a < image1.shape[1]) and (b < image1.shape[0]):
                        acc_space[b][a] += 1

circles = []
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if (v / r >= threshold and
            all(((x - xc) ** 2 + (y - yc) ** 2)**(1/2) > rmin/2 or
                abs(rc-r) > rmin/2 for xc, yc, rc in circles)):
        circles.append((x, y, r))
# parametry znalezionych okregow znajduja sie w circles

# tworzenie wykresow, do prezentacji wynikow, jak w przykladach

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(edges, cmap='gray')
axes[0].set_title('Uzyskane krawędzie obrazu')
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].imshow(acc_space, cmap='gray')
axes[1].set_title('Akumulacja w przestrzeni (a,b)')
axes[1].set_xlabel('a')
axes[1].set_ylabel('b')

axes[2].imshow(image1, cmap="gray")
axes[2].set_title('Obraz z nałożonymi wykrytymi okręgami')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')

# Dodawanie wykrytych okregow
for x, y, r in circles:
    circle = patches.Circle((x, y), r, edgecolor="red", facecolor="none", linewidth=1)
    axes[2].add_patch(circle)

axes[2].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
