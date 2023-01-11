#%% création de la vidéo test

import numpy as np
import cv2
import time
from scipy.integrate import odeint

# sys.setrecursionlimit(1000)

# Ce script à pour but de créer une vidéo animant un point rouge simulant la trajectoire d'un pendule simple.
# On la traite avec l'algorythme de traitement de vidéo pour comparer les données obtenues aux données attendues.

## équation du mouvement

g = 9.81                    # constante gravitationelle terrestre (m/s**2)
la = 0.2                    # coef de force de frottements (en kg*m/s)
lo = 0.20                   # longeur de la tige (en m) ; 39cm est arbitraire
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
dur = 5                      # durée de la video en secondes
framerate = 240               # nombre d'images par seconde
size = (1080, 1920)           # dimensions des frames de la vidéo créée
shutterSpeed = 1/150         # simule le temps d'ouverture de l'objectif
degrade = 5                  # discretise les positions du pendule pour le flou

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

    image = np.full( (size[1], size[0], 3), 255 )

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
    xt, yt = (x1-x0) /100, (y1-y0) /100     # On converti les valeurs de cm en m
    Xt.append(xt)
    Yt.append(yt)

ecarts_Xi = [abs(X[i]-Xt[i]) for i in range(len(X))]
ecarts_Yi = [abs(Y[i]-Yt[i]) for i in range(len(Y))]
ecarts_pos = [ ( ecarts_Xi[i]**2 + ecarts_Yi[i]**2 )**0.5 for i in range(len(ecarts_Xi))]

deltaxi = round(np.mean(ecarts_Xi)*1000, 2)
deltayi = round(np.mean(ecarts_Yi)*1000, 2)
deltapos = round(np.mean(ecarts_pos)*1000, 2)

print('avant lissage : ux =', deltaxi, 'mm ; uy =', deltayi, 'mm')
print('écart de position : upos = ', deltapos, 'mm')



#%% tests lissage par convolution
# On test la precison du lissage par fft

import numpy as np


def lissageconv (X):
    global b
    f = np.array([1, 2, 1])/4
    b = len(f)//2
    Xsm = np.convolve(X, f, mode='valid')
    return Xsm



Xlisse_conv, Ylisse_conv = lissageconv(Xt), lissageconv(Yt)
ux_conv, uy_conv = 0, 0
m = len(Xlisse_conv)
for i in range(m):
    ux_conv += Xlisse_conv[i] - X[i+b]
    uy_conv += Ylisse_conv[i] - Y[i+b]
ux_conv = round(ux_conv/m*1000, 2)
uy_conv = round(uy_conv/m*1000, 2)
upos_conv = round(((ux_conv)**2 + (uy_conv)**2)**0.5, 2)

print('\n___convolution___')
print('après lissage par convolution : ux =', ux_conv, 'mm ; uy =', uy_conv, 'mm')
print('écart de position : upos = ', upos_conv, 'mm')



#%% Visualisation des résultats

def deriv (X):
    global T
    D = []
    for i in range(1,len(X)-1):
        D.append( (X[i+1]-X[i-1])/(T[i+1]-T[i-1]) )
    return D

dX = deriv(X)

# plt.figure('test')
# plt.plot(T[1:-1], dX, color='red', label='dX')
# plt.plot(T[11:-11], dXl[10:-10], color='blue', label='dX lissé')
# plt.legend()
# plt.grid()
# plt.show()


plt.figure('Résultats')
plt.clf()

plt.subplot(1, 2, 1)
plt.plot(T, X, label='X',color='green')
# plt.plot(T, Xd, label='X discretisé', color='Blue')
# plt.plot(T, Xt, label='X traité', color='red')
plt.plot(T[b:-b], Xlisse_conv, label='X lissé',color='orange')
plt.xlabel('temps (en s)')
plt.ylabel('x (en m)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(T, Y, label='Y', color='green')
# plt.plot(T, Yd, label='Y discretisé', color='Blue')
#plt.plot(T, Yt, label='Y traité', color='red')
plt.plot(T[b:-b], Ylisse_conv, label='Y lissé', color='orange')
plt.xlabel('temps (en s)')
plt.ylabel('y (en m)')
plt.legend()
plt.grid()

# plt.figure('Résultats éclatés')
# plt.clf()
#
# donnees = [X, Y, Xd, Yd, Xt, Yt]
# Labels = ['X', 'Y', 'X discretisé', 'Y discretisé', 'X traité', 'Y traité']
# colors = ['green', 'blue', 'red']
# k = 0
# for i in range (1, 7):
#     plt.subplot(3, 2, i)
#     plt.plot(T, donnees[i-1], label=Labels[i-1], color=colors[k])
#     plt.legend()
#     plt.grid()
#     if i % 2 == 0 and k < 2 :
#         k += 1

plt.show()