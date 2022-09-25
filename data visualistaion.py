import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def str2float(N:str):
    if N == '' or N == '\n':
        return 0.0
    else :
        return float(N)

dos = open('/Users/pabloarb/Desktop/data/test 2/csv/positions objets.csv', 'r')

lines = dos.readlines()
T = []
X0 = []
Y0 = []
X1 = []
Y1 = []

for line in lines[5:] :

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

def lissage (X, th):
    newX = [X[0], X[1]]
    for i in range(1, len(X)-1):
        if np.abs(X[i+1]-newX[i]) > th :
            # prediction = X[i] + newX[i]-newX[i-1]
            # x = (X[i+1] + prediction)/2
            # newX.append(x)
            newX.append(newX[i])
        else :
            newX.append(X[i+1])
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
c = 30

Xrel = [ X0[i]-X1[i] for i in range( len(X0) ) ]
Yrel = [ Y0[i]-Y1[i] for i in range( len(Y0) ) ]

plt.figure("relatif")
plt.clf()
plt.plot(T, Xrel, label='X', color='blue')
plt.xlabel('Temps(en s)')
plt.ylabel('X(en cm) ')
plt.grid()
plt.legend()

plt.figure("relatif dérivé")
plt.clf()
plt.plot(T, lissage(deriv(Xrel), c), label='dX/dt', color='red')
plt.xlabel('Temps(en s)')
plt.ylabel('X(en cm) ')
plt.grid()
plt.legend()


plt.figure("figure de phase relatif")
plt.clf()
plt.plot(Xrel, lissage(deriv(Xrel), c), label='X relatif', color='green')
plt.xlabel('X')
plt.ylabel('dX/dt')
plt.grid()
plt.legend()

plt.show()