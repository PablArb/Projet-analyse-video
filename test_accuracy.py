#%% import des modules

import numpy as np
import cv2
import time
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Ce script à pour but de créer une vidéo animant un point rouge simulant la trajectoire d'un pendule simple.
# On la traite avec l'algorythme de traitement de vidéo pour comparer les données obtenues aux données attendues.

## équation du mouvement

# paramètres pendule
g = 9.81                    # constante gravitationelle terrestre (m/s**2)
la = 0.0                    # coef de force de frottements (en kg*m/s)
lo = 0.25                   # longeur de la tige (en m)
theta0 = np.pi/9            # angle initial (en rad)
v0 = 0                      # vitesse initiale du pendule (en m/s)

# paramètres vidéo
name = 'accuracy test vidéo'
dur = 10                    # durée de la video en secondes
framerate = 120             # nombre d'images par seconde
size = (720, 1280)          # dimensions des frames de la vidéo créée
shutterSpeed = 1/300        # simule le temps d'ouverture de l'objectif
degrade = 10                # discretise les positions du pendule pour le flou
blanc = 2

# configs :
# normal : (1080, 1920) ; 30 fps
# slowmo : (720, 1280) ; 120 fps

N = dur * framerate # nombre de valeur de theta que l'on veut calculer
T = np.linspace(0, dur, N) # temps associés à ces valeures de theta
rayon = int(size[1]/60) # rayon du point dessiné sur les frames
marge = (rayon + 500)*size[1]/2000 # écart entre le centre des points et les bordures de l'image

# équation differentielle vérifiée par theta
def dSdt (S, t):
    global g, la, lo
    theta, v = S
    return (v, -la * v - g/lo * np.sin(theta))

# projetés sur les axes cartésiens horizontaux et verticaux
def pos_xy (theta):
    global lo
    x = lo * np.sin(theta)
    y = lo * np.cos(theta)
    return x, y

# calcul des valeurs de theta (discretisation de l'équation)
def Theta_dico():
    global T
    dico = {}
    res = odeint (dSdt, (theta0, v0), T)
    assert len(res) == len(T)
    for i in range(len(res)) :
        dico[T[i]]=res[i]
    return dico

# On veut une vidéo potable quelle que soient les dimensions choisies pour la vidéo pour la vidéo.
# On centre le pendule.
def mise_a_echelle (x, y):
    global size, lo, marge, echelle
    echelle = (size[1]-marge) / lo
    x = int(size[0]/2 + x * echelle)
    y = int(y * echelle + marge/2)
    return x, y

# On trace le point rouge aux coordonnées indiquées.
def point(image, x, y):
    global size, rayon, degrade
    x, y = int(x), int(y)
    for i in range (x-rayon, x+rayon):
        for j in range (y-rayon, y+rayon):
            if 0<=i<size[0] and 0<=j<size[1] :
                if ( (i-x)**2 + (j-y)**2 )**0.5 < rayon :
                    image[j][i][0] -= int(255/degrade)
                    image[j][i][1] -= int(255/degrade)
    return np.uint8(image)


X = []
Xd = []
Y = []
Yd = []

s = time.time()
Theta = Theta_dico() # contient [Theta, Thetapoint] pour chaque t dans T
e = time.time()
dur = round(e-s, 5)
print('\nTemps mis pour calculer Theta(t) : ' + str(dur) + 's')

# On crée la vidéo.

Ti = time.time()
path = '/Users/pabloarb/Desktop/'+ name +'.mp4'
format = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(path, format, framerate, size)
for t in T:

    # image = np.uint8(np.full( (size[1], size[0], 3), 255 ))

    if not t in T[N//2:N//2+blanc]:
        dT = np.linspace(t-shutterSpeed/2, t+shutterSpeed/2, degrade)
        dTheta = odeint(dSdt, Theta[t], dT)[:,0]
        for i in range(len(dTheta)):
            x, y = pos_xy(dTheta[i])
            xe, ye = mise_a_echelle(x, y)
            # image = point(image, xe, ye)
            # image = point(image, size[0]//2, marge//2)
            if i == len(dT)//2 :
                X.append(x)
                Y.append(y)
                Xd.append(xe)
                Yd.append(ye)
    else :
        X.append(x)
        Y.append(y)
    # out.write(image)

    progr = round(t/T[-1] * 100, 1)
    print('\rcréation de la vidéo : ' + str(progr) + '%', end='')

# out.release()
print('\nTemps mis pour créer la vidéo : ' + str( round(time.time()-Ti, 2) ) + 's')



#%% Récupération des données
# Après avoir effectuer le traitement de la vidéo on extrait les mesures effectuées

name = 'accuracy test vidéo'

def str2float(N:str):
    if N == '' or N == '\n':
        print(1)
        return 0.0
    else :
        return float(N)

dos = open('/Users/pabloarb/Desktop/mes exp TIPE/data video/' + name + '.mp4/csv/positions objets.csv', 'r')
lines = dos.readlines()

Xt = []
Yt = []
for line in lines[1:] :
    line = line.split(',')
    x0, y0 = str2float(line[2]), str2float(line[3])
    x1, y1 = str2float(line[4]), str2float(line[5])
    xt, yt = (x1-x0), (y1-y0)
    Xt.append(xt/100)
    Yt.append(yt/100)


ecarts_Xi = [(X[i]-Xt[i])*1000 for i in range(len(X))]
ecarts_Yi = [(Y[i]-Yt[i])*1000 for i in range(len(Y))]
ecarts_pos = [ ( ecarts_Xi[i]**2 + ecarts_Yi[i]**2 )**0.5 for i in range(len(ecarts_Xi))]

deltaxi = round(np.mean(np.abs(ecarts_Xi)), 3)
stdxi = round(np.std(ecarts_Xi, ddof=1), 3)

deltayi = round(np.mean(np.abs(ecarts_Yi)), 3)
stdyi = round(np.std(ecarts_Yi, ddof=1), 3)

deltapos = round(np.mean(ecarts_pos), 3)
stdpos = round(np.std(ecarts_pos, ddof=1), 3)


print('avant lissage :\t ux =', deltaxi, 'mm\t\t stdx =', stdxi, 'mm')
print('\t\t\t\t uy =', deltayi, 'mm\t\t stdy =', stdyi, 'mm')
print('\t\t\t\t upos =', deltapos, 'mm\t\t stdy =', stdpos, 'mm')


plt.figure(1)
plt.clf()
plt.scatter(Xt, ecarts_Xi)
plt.grid()
plt.show()

plt.figure(2)
plt.clf()
plt.plot(T, X)
plt.plot(T, Xt)
plt.grid()
plt.show()

plt.figure(3)
plt.clf()
plt.plot(T, Y)
plt.plot(T, Yt)
plt.grid()
plt.show()

#%% tests lissage par convolution
# On test la precison du lissage par fft

import numpy as np

def lissageconv (X):
    global b
    a = [1, 2, 4, 2, 1]
    f = np.array(a)/np.sum(a)
    b = len(f)
    Xsm = np.convolve(X, f, mode='same')
    return Xsm

X_sm, Y_sm = lissageconv(Xt), lissageconv(Yt)

ecarts_Xi_sm = [(X[i]-X_sm[i])*1000 for i in range(b, len(X)-b)]
ecarts_Yi_sm = [(Y[i]-Y_sm[i])*1000 for i in range(b, len(Y)-b)]
ecarts_pos_sm = [ ( ecarts_Xi_sm[i]**2 + ecarts_Yi_sm[i]**2 )**0.5 for i in range(len(ecarts_Xi_sm))]

deltaxi_sm = round(np.mean(np.abs(ecarts_Xi_sm)), 3)
stdxi_sm = round(np.std(ecarts_Xi_sm, ddof=1), 3)

deltayi_sm = round(np.mean(np.abs(ecarts_Yi_sm)), 3)
stdyi_sm = round(np.std(ecarts_Yi_sm, ddof=1), 3)

deltapos_sm = round(np.mean(ecarts_pos_sm), 3)
stdpos_sm = round(np.std(ecarts_pos_sm, ddof=1), 3)


print('convolution :\t ux =', deltaxi_sm, 'mm\t\t stdx =', stdxi_sm, 'mm')
print('\t\t\t\t uy =', deltayi_sm, 'mm\t\t stdy =', stdyi_sm, 'mm')
print('\t\t\t\t upos =', deltapos_sm, 'mm\t\t stdy =', stdpos_sm, 'mm')


plt.figure(4)
plt.clf()
plt.scatter(Xt[b:-b], ecarts_Xi_sm)
plt.grid()
plt.show()

plt.figure(5)
plt.clf()
plt.plot(T, X, '-')
plt.plot(T[b:-b], X_sm[b:-b], '-')
plt.grid()
plt.show()

plt.figure(6)
plt.clf()
plt.plot(T, Y, '-')
plt.plot(T[b:-b], Y_sm[b:-b], '-')
plt.grid()
plt.show()

#%% test modélisation

# def Xfct(t, A, w, tau, phi):
#     return A*np.cos(w*t+phi)*np.exp(-t/tau)
#
# init = (1, 2, 10, -0.033)
# A, w, tau, phi = curve_fit(Xfct, T, X, init)[0]
# X_md = [Xfct(t, A, w, tau, phi) for t in T]


def Xfct(t, A, w, phi, offset):
    return A*np.cos(w*t+phi)+offset

init = (0, 6, 0, 0)
A, w, phi, offset = curve_fit(Xfct, T, X, init)[0]
X_md = [Xfct(t, A, w, phi, offset) for t in T]


Y_md = [(lo**2-x**2)**0.5 for x in X_md]


ecarts_Xi_md = [(X[i]-X_md[i])*1000 for i in range(len(X))]
ecarts_Yi_md = [(Y[i]-Y_md[i])*1000 for i in range(len(Y))]
ecarts_pos_md = [ ( ecarts_Xi_md[i]**2 + ecarts_Yi_md[i]**2 )**0.5 for i in range(len(ecarts_Xi_md))]

deltaxi_md = round(np.mean(np.abs(ecarts_Xi_md)), 3)
stdxi_md = round(np.std(ecarts_Xi_md, ddof=1), 3)

deltayi_md = round(np.mean(np.abs(ecarts_Yi_md)), 3)
stdyi_md = round(np.std(ecarts_Yi_md, ddof=1), 3)

deltapos_md = round(np.mean(ecarts_pos_md), 3)
stdpos_md = round(np.std(ecarts_pos_md, ddof=1), 3)


print('modelisation :\t ux =', deltaxi_md, 'mm\t\t stdx =', stdxi_md, 'mm')
print('\t\t\t\t uy =', deltayi_md, 'mm\t\t stdy =', stdyi_md, 'mm')
print('\t\t\t\t upos =', deltapos_md, 'mm\t\t stdy =', stdpos_md, 'mm')


N = 50
p = max((X)-min(X))/(N-1)
X_hist = np.linspace(min(X), max(X), N)

Y_hist = np.zeros(len(X_hist))
s = len(X)
for i in range(len(X_md)):
    for j in range(len(X_hist)):
        if X_hist[j]-p < X_md[i] < X_hist[j]+p:
            Y_hist[j] += abs(X[i]-X_md[i])

plt.figure(7)
plt.clf()
plt.plot(X_hist, Y_hist)
plt.grid()
plt.show()


plt.figure(8)
plt.clf()
plt.plot(T, X, '-')
plt.plot(T, X_md, '-')
plt.grid()
plt.show()

plt.figure(9)
plt.clf()
plt.plot(T, Y, '-')
plt.plot(T, Y_md, '-')
plt.grid()
plt.show()


