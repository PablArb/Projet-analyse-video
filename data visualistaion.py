import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def str2float(N:str):
    if N == '' or N == '\n':
        return 0.0
    else :
        return float(N)

dos = open('Desktop/Test/video test/csv/positions objets.csv', 'r')

lines = dos.readlines()
T = []
X0 = []
Y0 = []
X1 = []
Y1 = []
X2 = []
Y2 = []

for line in lines[1:] :

    line = line.split(',')
    time = str2float(line[1])
    x0, y0 = str2float(line[2]), str2float(line[3])
    x1, y1 = str2float(line[4]), str2float(line[5])
    x2, y2 = str2float(line[6]), str2float(line[7])
    T.append(time)
    X0.append(x0)
    Y0.append(y0)
    X1.append(x1)
    Y1.append(y1)
    X2.append(x2)
    Y2.append(y2)

Xrel = [ X0[i]-X1[i] for i in range( len(X0) ) ]
Yrel = [ Y0[i]-Y1[i] for i in range( len(Y0) ) ]


plt.figure('position')
plt.clf()
ax = plt.axes(projection='3d')

ax.plot(T, X0, Y0, label='obj0')
ax.plot(T, X1, Y1, label='obj1')
ax.plot(T, X2, Y2, label='obj2')

plt.legend()
plt.show()