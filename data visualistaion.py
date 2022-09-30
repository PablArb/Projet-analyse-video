import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import rfft


def str2float(N:str):
    if N == '' or N == '\n':
        return 0.0
    else :
        return float(N)

video = 'test C - pendule sur tour long'
dos = open('/Users/pabloarb/Desktop/data/' + video + '/csv/positions objets.csv', 'r')

lines = dos.readlines()
T = []
X0 = []
Y0 = []
X1 = []
Y1 = []

for line in lines[1:] :

    line = line.split(',')
    time = str2float(line[1])

    x0, y0 = str2float(line[2]), str2float(line[3])
    x1, y1 = str2float(line[4]), str2float(line[5])

    T.append(time)
    X0.append(x0)
    Y0.append(y0)
    X1.append(x1)
    Y1.append(y1)

def deriv (X):
    global T
    D = []
    for i in range(len(X)-1):
        D.append( (X[i+1]-X[i])/(T[i+1]-T[i]) )
    D.append ( (X[-1]-X[-2]) / (T[-1]-T[-2]))
    return D

def lissage (X, fc):
    global T
    pi  = np.pi
    N   = len(X)
    fe  = len(T) / (T[-1]-T[0])

    TF = rfft(X)
    Normalisation = np.array ([1/ N ]+[2/ N for k in range ( N //2)])
    Ampl = np.abs( Normalisation * TF )
    Phase = np.angle(TF)
    freq = np.array ([ k * fe / N for k in range ( N//2 +1)])

    F = []
    i = 0
    while freq[i] < fc :
        F.append(freq[i])
        i += 1

    newX = []
    for t in T:
        xi = 0
        for i in range ( len(F) ):
            A = Ampl[i]
            w = 2*pi*F[i]
            phi = Phase[i]
            xi += A * np.cos( w * t + phi )
        newX.append(xi)
    return newX


## demo

Xrel = [ X0[i]-X1[i] for i in range( len(X0) ) ]
Yrel = [ Y0[i]-Y1[i] for i in range( len(Y0) ) ]
zero = [0]*len(Xrel)

plt.figure("demonstration")
plt.clf()
ax = plt.axes(projection='3d')
ax.plot(T, zero, zero, label='obj0')
ax.plot(T, Xrel, Yrel, label='obj1')
plt.legend()

plt.show()




## data analyse

#on def un seuil de pente pour le lissage
fc = 3

Xrel = [ X0[i]-X1[i] for i in range( len(X0) ) ]
Yrel = [ Y0[i]-Y1[i] for i in range( len(Y0) ) ]
Xlisse = lissage(Xrel, fc)
# Ylisse = lissage(Yrel, fc)
dX = deriv(Xrel)
# dY = deriv(Yrel)
dXlisse = deriv(Xlisse)
# dYlisse = deriv(Ylisse)

plt.figure("relatif")
plt.clf()
plt.plot(T, Xrel,   label='X',       color='blue')
plt.plot(T[10:-10], Xlisse[10:-10], label='X lissé', color='orange')
plt.xlabel('Temps(en s)')
plt.ylabel('X(en cm) ')
plt.grid()
plt.legend()

plt.figure("relatif dérivé")
plt.clf()
plt.plot(T, dX,      label='dX/dt',       color='blue')
plt.plot(T[10:-10], dXlisse[10:-10], label='dX/dt lissé', color='orange')
plt.xlabel('Temps(en s)')
plt.ylabel('X(en cm) ')
plt.grid()
plt.legend()

plt.figure("figure de phase relatif")
plt.clf()
plt.plot(Xlisse[10:-10], dXlisse[10:-10], label='X relatif', color='green')
plt.xlabel('X')
plt.ylabel('dX/dt')
plt.grid()
plt.legend()

plt.show()