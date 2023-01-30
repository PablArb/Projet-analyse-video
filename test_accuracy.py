#%% création de la vidéo test

import numpy as np
import cv2
import time
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# sys.setrecursionlimit(1000)

# Ce script à pour but de créer une vidéo animant un point rouge simulant la trajectoire d'unx pendule simple.
# On la traite avec l'algorythme de traitement de vidéo pour comparer les données obtenues aux données attendues.

## équation du mouvement

g = 9.81                    # constante gravitationelle terrestre (m/s**2)
la = 0.2                    # coef de force de frottements (en kg*m/s)
lo = 0.39                   # longeur de la tige (en m) ; 39cm est arbitraire
theta0 = np.pi/9            # angle initial (en rad)
v0 = 0                      # vitesse initiale du pendule (en m/s)

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


# création de la vidéo de test
name = 'accuracy test vidéo'
dur = 3                      # durée de la video en secondes
framerate = 120              # nombre d'images par seconde
size = (720, 1280)           # dimensions des frames de la vidéo créée
shutterSpeed = 1/300         # simule le temps d'ouverture de l'objectif
degrade = 5                  # discretise les positions du pendule pour le flou
blanc = 2

rayon = int(size[1]/60) # rayon du point dessiné sur les frames
marge = (rayon + 500)*size[1]/2000 # écart entre le centre des points et les bordures de l'image

# configs tel Edouard :
# 4K : (2160, 3840) ; 60 fps
# SM : (1080, 1920) ; 240 fps

# configs tel Pablo :
# normal : (1080, 1920) ; 30 fps
# slowmo : (720, 1280) ; 120 fps

N = dur * framerate # nombre de valeur de theta que l'on veut calculer
T = np.linspace(0, dur, N) # temps associés à ces valeures de theta


# calcul des valeurs de theta (discretisation de l'équation)
def Theta_dico():
    global T
    dico = {}
    res = odeint (dSdt, (theta0, v0), T)
    assert len(res) == len(T)
    for i in range(len(res)) :
        dico[T[i]]=res[i]
    return dico

s = time.time()
Theta = Theta_dico() # contient [Theta, Thetapoint] pour chaque t dans T
e = time.time()
dur = round(e-s, 5)
print('\nTemps mis pour calculer Theta(t) : ' + str(dur) + 's')


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

# On crée la vidéo.

Ti = time.time()
path = '/Users/pabloarb/Desktop/'+ name +'.mp4'
format = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(path, format, framerate, size)
for t in T:

    image = np.uint8(np.full( (size[1], size[0], 3), 255 ))

    if not t in T[N//2:N//2+blanc]:
        dT = np.linspace(t, t+shutterSpeed, degrade)
        dTheta = odeint(dSdt, Theta[t], dT)[:,0]
        for i in range(len(dTheta)):
            x, y = pos_xy(dTheta[i])
            xe, ye = mise_a_echelle(x, y)
            image = point(image, xe, ye)
            image = point(image, int(size[0]/2), int(marge/2))
            if i == int(len(dT)/2):
                X.append(x)
                Y.append(y)
                Xd.append(xe)
                Yd.append(ye)
    out.write(image)

    progr = round(t/T[-1] * 100, 1)
    print('\rcréation de la vidéo : ' + str(progr) + '%', end='')

out.release()
print('\nTemps mis pour créer la vidéo : ' + str( round(time.time()-Ti, 2) ) + 's')



#%% Récupération des données
# Après avoir effectuer le traitement de la vidéo on extrait les mesures effectuées

name = 'accuracy test vidéo'

def str2float(N:str):
    if N == '' or N == '\n':
        return 0.0
    else :
        return float(N)

dos = open('/Users/pabloarb/Desktop/mes exp TIPE/data video/' + name + '.mp4/csv/positions objets.csv', 'r')
lines = dos.readlines()

Xt = []
Yt = []
for line in lines[1:] :
    line = line.split(',')
    time = str2float(line[1])
    x0, y0 = str2float(line[2]), str2float(line[3])
    x1, y1 = str2float(line[4]), str2float(line[5])
    xt, yt = (x1-x0), (y1-y0)
    Xt.append(xt/100)
    Yt.append(yt/100)

ecarts_Xi = [abs(X[i]-Xt[i])*1000 for i in range(len(X))]
ecarts_Yi = [abs(Y[i]-Yt[i])*1000 for i in range(len(Y))]
ecarts_pos = [ ( ecarts_Xi[i]**2 + ecarts_Yi[i]**2 )**0.5 for i in range(len(ecarts_Xi))]

deltaxi = round(np.mean(ecarts_Xi), 3)
deltayi = round(np.mean(ecarts_Yi), 3)
deltapos = round(np.mean(ecarts_pos), 3)

print('avant lissage : ux =', deltaxi, 'mm ; uy =', deltayi, 'mm')
print('écart de position : upos = ', deltapos, 'mm')

plt.figure(1)
plt.clf()
plt.scatter(Xt, ecarts_Xi)
plt.show()

plt.figure(2)
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

ecarts_Xi_sm = [abs(X[i]-X_sm[i])*1000 for i in range(b, len(X)-b)]
ecarts_Yi_sm = [abs(Y[i]-Y_sm[i])*1000 for i in range(b, len(Y)-b)]
ecarts_pos_sm = [ ( ecarts_Xi_sm[i]**2 + ecarts_Yi_sm[i]**2 )**0.5 for i in range(len(ecarts_Xi_sm))]

deltaxi_sm = round(np.mean(ecarts_Xi_sm), 3)
deltayi_sm = round(np.mean(ecarts_Yi_sm), 3)
deltapos_sm = round(np.mean(ecarts_pos_sm), 3)

print('après lissage par convolution : ux =', deltaxi_sm, 'mm ; uy =', deltayi_sm, 'mm')
print('écart de position : upos = ', deltapos_sm, 'mm')


plt.figure(3)
plt.clf()
plt.scatter(Xt[b:-b], ecarts_Xi_sm)
plt.show()


plt.figure(4)
plt.clf()
plt.plot(T, X, 'o-')
plt.plot(T[b:-b], X_sm[b:-b], 'o-')
plt.grid()
plt.show()

#%% test modélisation

def Xfct(t, A, w, tau, phi):
    return A*np.cos(w*t+phi)*np.exp(-t/tau)

init = (0.10, 5, 10, 0)
A, w, tau, phi = curve_fit(Xfct, T, X, init1)[0]
X_md = [Xfct(t, A1, w1, tau1, phi1) for t in T]

Y_md = [(0.39**2-x**2)**0.5 for x in X_md]


ecarts_Xi_md = [abs(X[i]-X_md[i])*1000 for i in range(len(X))]
ecarts_Yi_md = [abs(Y[i]-Y_md[i])*1000 for i in range(len(Y))]
ecarts_pos_md = [ ( ecarts_Xi_md[i]**2 + ecarts_Yi_md[i]**2 )**0.5 for i in range(len(ecarts_Xi_md))]

deltaxi_md = round(np.mean(ecarts_Xi_md), 3)
deltayi_md = round(np.mean(ecarts_Yi_md), 3)
deltapos_md = round(np.mean(ecarts_pos_md), 3)

print('après lissage par convolution : ux =', deltaxi_md, 'mm ; uy =', deltayi_md, 'mm')
print('écart de position : upos = ', deltapos_md, 'mm')

N = 15
p = max((X)-min(X))/(N-1)
X_hist = np.linspace(min(X), max(X), N)

Y_hist = np.zeros(len(X_hist))
s = len(X)
for i in range(len(X_md)):
    for j in range(len(X_hist)):
        if X_hist[j]-p < X_md[i] < X_hist[j]+p:
            Y_hist[j] += abs(X[i]-X_md[i])

plt.figure(5)
plt.clf()
plt.plot(X_hist, Y_hist)
plt.grid()
plt.show()


plt.figure(6)
plt.clf()
plt.plot(T, X)
plt.plot(T, X_md)
plt.grid()
plt.show()


