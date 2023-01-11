import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


video = 'test lego tour A basique.mp4' # à renseigner

videoData = '/Users/pabloarb/Desktop/mes exp TIPE/data video/' + video + '/csv/positions objets.csv'

dos = open(videoData, 'r')
lines = dos.readlines()

def str2float(N:str):
    if N == '' or N == '\n':
        return 0.0
    else :
        return float(N)

T = []
n = int((len(lines[0].split(','))-2)/2)

for i in range(n):
    exec(f'X{i}, Y{i} = [], []')

for line in lines[1:] :
    line = line.split(',')
    time = str2float(line[1])
    T.append(time)
    for i in range(n):
        exec(f'x{i}, y{i} = str2float(line[(i+1)*2])/100, str2float(line[(i+1)*2+1])/100')
        exec(f'X{i}.append(round(x{i}, 7))')
        exec(f'Y{i}.append(round(y{i}, 7))')
T = np.array(T)
for i in range(n):
    exec(f"X{i}, Y{i} = np.array(X{i}), np.array(Y{i})")

def gaussian(x, mu, sig):
    return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

mu = 0
sig = 1
x = np.linspace(-2*sig, 2*sig, 5)
filtre = gaussian(x, mu, sig)
f_moy = filtre/np.sum(filtre)

for i in range(n):
    exec(f"Xs{i} = np.convolve(f_moy, X{i}, 'same')")
    exec(f"Ys{i} = np.convolve(f_moy, Y{i}, 'same')")

def deriv (X):
    global T, k
    D = []
    k = 2
    for i in range(k,len(X)-k):
        D.append( (X[i+k]-X[i-k])/(T[i+k]-T[i-k]) )
    return D

## demo

plt.figure("demonstration")
plt.clf()
ax = plt.axes(projection='3d')

for k in range (n):
    exec(f"Xp{k} = X{k} - X{n-1}")
    exec(f"Yp{k} = Y{n-1} - Y{k}")
    exec(f"ax.plot(T, Xp{k}, Yp{k}, label='obj{k}')")

ax.set_xlabel('temps (en s)')
ax.set_ylabel('X')
ax.set_zlabel('Y')
ax.legend()
plt.show()


## data visu détaillée


plt.figure("relatif")
plt.clf()

T2 = T[20:-20]
for i in range(1, 7):

    exec(f'X, Xs = X{i}[20:-20], Xs{i}[20:-20]')
    # exec(f'Y = Y{i}[20:-20], Ys{i}[20:-20]')

    exec(f"leg1 = 'x{i}'")
    exec(f"leg2 = 'x{i} smoothed'")
    # exec(f"leg3 = 'y{i}'")
    # exec(f"leg4 = 'y{i} smoothed'")

    plt.subplot(3, 2, i)

    plt.plot(T2, X, label=leg1, color='blue')
    plt.plot(T2, Xs, label=leg2, color='red')

    # plt.plot(T2, Y, label=leg3, color='green')
    # plt.plot(T2, Ys, label=leg3, color='orange')

    plt.xlabel('Temps (en s)')
    plt.ylabel('signal (en m) ')
    plt.grid()
    plt.legend()


plt.show()

