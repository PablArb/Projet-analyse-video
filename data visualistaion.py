import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import rfft

def str2float(N:str):
    if N == '' or N == '\n':
        return 0.0
    else :
        return float(N)

video = 'test Z'
dos = open('/Users/pabloarb/Desktop/data/' + video + '/csv/positions objets.csv', 'r')

lines = dos.readlines()

T = []
n = int((len(lines[0].split(','))-2)/2)

for i in range(n):
    exec(f'X{i}, Y{i} = [], []')

for line in lines[1:] :

    line = line.split(',')
    time = str2float(line[1])

    T.append(time)

    for i in range(n):
        exec(f'x{i}, y{i} = str2float(line[(i+1)*2]), str2float(line[(i+1)*2+1])')
        exec(f'X{i}.append(x{i})')
        exec(f'Y{i}.append(y{i})')

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

plt.subplot(2, 2, 1)
plt.plot(T, Xrel,   label='X',       color='blue')
plt.plot(T[10:-10], Xlisse[10:-10], label='X lissé', color='orange')
plt.xlabel('Temps(en s)')
plt.ylabel('X(en cm) ')
plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(T, dX,      label='dX/dt',       color='blue')
plt.plot(T[10:-10], dXlisse[10:-10], label='dX/dt lissé', color='orange')
plt.xlabel('Temps(en s)')
plt.ylabel('X(en cm) ')
plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(Xrel, dX, label='phase non lissée', color='blue')
plt.xlabel('X')
plt.ylabel('dX/dt')
plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(Xlisse[10:-10], dXlisse[10:-10], label='phase lissée', color='orange')
plt.xlabel('X')
plt.ylabel('dX/dt')
plt.grid()
plt.legend()

plt.show()

## special

plt.figure("demonstration")
plt.clf()
ax = plt.axes(projection='3d')
ax.plot(T, X0, Y0, label='obj0')
plt.legend()

plt.show()