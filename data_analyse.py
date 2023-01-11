## importation des modules utiles
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


## vidéo à étudier
video = 'test lego tour A basique.mp4'



settings = '/Users/pabloarb/Desktop/mes exp TIPE/data video/' + video + '/csv/settings.csv'

## récupération des données

def str2float(N:str):
    '''
    convertis la valeure passée en argument en string
    avec certains cas particuliers propre à la mise en forme
    effectuée lors de la mesure.
    '''
    if N == '' or N == '\n':
        return 0.0
    else :
        return float(N)



videoData = '/Users/pabloarb/Desktop/mes exp TIPE/data video/' + video + '/csv/positions objets.csv'

dos = open(videoData, 'r')
lines = dos.readlines()
dos.close()

T = []
n = int((len(lines[0].split(','))-2)/2)
Xs, Ys = [], []

for i in range(n):
    exec(f'X{i}, Y{i} = [], []')

for line in lines[1:] :
    line = line.split(',')
    time = str2float(line[1])
    T.append(time)
    for i in range(n):
        exec(f'x{i}, y{i} = str2float(line[(i+1)*2]), str2float(line[(i+1)*2+1])')
        exec(f'X{i}.append(round(x{i}, 7))')
        exec(f'Y{i}.append(round(y{i}, 7))')
T = np.array(T)
for i in range(n):
    exec(f"X{i}, Y{i} = np.array(X{i}), np.array(Y{i})")
    exec(f'Xs.append(X{i})')
    exec(f'Ys.append(Y{i})')



dos = open(settings, 'r')
lines = dos.readlines()
dos.close()

# La récupération du fps est utile pour effectuer une analyse de fourier.
s = 'Framerate'
l = len(s)
for line in lines :
    if line[:l] == s:
        line = line[:-1]
        complete, i = False, -1
        while not complete and i > -10:
            try :
                framerate = float(line[i:])
                i-=1
            except ValueError:
                complete = True
                framerate = float(line[i+1:])

## analyse préalable des données
# L'objectif est ici d'effectuer la modelisation pour une des points repères
# pour determiner l'intervalle temporel le plus adapté a l'étude.

XStudied = X6

def indice (el, L):
    assert L[0]<=el<=L[-1]
    i=0
    while L[i] < el :
        i += 1
    return i + 1

def sinus(t, A, w, phi):
    return A * np.sin(w*t + phi) # Je ne comprends pas le 12pi mais ca marche.

Nfit = 10000 # Nombre de point contenus dans le signal créé.
l1, l2 = 4.3, 6.7 # bornes temporelles entre lesquelles l'étude est intéressante.

# Definition du domaine de validité du modèle
ilim1 = indice (l1, T)
ilim2 = indice (l2, T)

# initial = (amplitude, angular frequency, phase)
initial = (0, 36, 0)

# Definition du modèle
A, w, phi = curve_fit(sinus, T[ilim1:ilim2], XStudied[ilim1:ilim2], initial)[0]
while phi < 0 : phi += 2*np.pi

# création du signal modèle
Tfit = np.linspace(T[ilim1], T[ilim2], Nfit)
Xfit = sinus(Tfit, A, w, phi)

# ajustement pour visualisation
offset = np.mean(XStudied[ilim1:ilim2])
XStudied -= offset
limx = [1.2*A, -1.2*A]
limit1 = [T[ilim1] for t in limx]
limit2 = [T[ilim2] for t in limx]


## affichage des résulats préliminaire

plt.figure('visu')
plt.clf()
plt.title('modélisation pour '+str(Nfit)+' points')

plt.plot(limit1, limx, '--', color='red', linewidth=3, zorder=5)
plt.plot(limit2, limx, '--', color='red', linewidth=3, zorder=5)

plt.plot(T, XStudied, label='X')
plt.plot(Tfit, Xfit, label='modèle associé à X : '+str(round(A,3))+'sin('+str(round(w, 3))+'t+'+str(round(phi,3))+')')

plt.xlabel('temps (en s)')
plt.ylabel('signal (en cm)')
plt.legend()

plt.grid()
plt.show()



## analyse des données

class Tower ():
    def __init__(self, id, time, timeInterval, markercount):
        self.id = id
        self.points = []
        self.time = time
        self.ValidityInterval = timeInterval
        self.markercount = markercount

class Point ():
    def __init__(self, id, nat, posMes, posMod, Y):
        self.altitude = Y
        self.id = id
        self.nature = nat
        self.posMes = posMes
        self.posMod = posMod

class SinModel ():
    def __init__(self, A, w, phi):
        self.amplitude = A
        self.pulsation = w
        self.dephasage = phi

    def sinus(t, A, w, phi):
        return A * np.sin(w*t + phi)

def str2float(N:str):
    '''
    convertis la valeure passée en argument en string
    avec certains cas particuliers propre à la mise en forme
    effectuée lors de la mesure.
    '''
    if N == '' or N == '\n':
        return 0.0
    else :
        return float(N)

def indice (t, Temps):
    assert Temps[0]<=t<=Temps[-1]
    i=0
    while Temps[i] < t :
        i += 1
    return i + 1

def getData (video):

    videoData = '/Users/pabloarb/Desktop/mes exp TIPE/data video/' + video + '/csv/positions objets.csv'

    dos = open(videoData, 'r')
    lines = dos.readlines()
    dos.close()

    T = []
    n = int((len(lines[0].split(','))-2)/2)
    Xs, Ys = [], []

    for i in range(n):
        exec(f'X{i}, Y{i} = [], []')

    for line in lines[1:] :
        line = line.split(',')
        time = str2float(line[1])
        T.append(time)
        for i in range(n):
            exec(f'x{i}, y{i} = str2float(line[(i+1)*2]), str2float(line[(i+1)*2+1])')
            exec(f'X{i}.append(round(x{i}, 7))')
            exec(f'Y{i}.append(round(y{i}, 7))')
    T = np.array(T)
    for i in range(n):
        exec(f"X{i}, Y{i} = np.array(X{i}), np.array(Y{i})")
        exec(f'Xs.append(X{i})')
        exec(f'Ys.append(Y{i})')

    return T, Xs, Ys, n

def analysis(video, timeInterval):

    initial = (0, 36, 0)
    T, Xs, Ys, n = getData(video)
    tour = Tower(video, T, timeInterval, n)

    l1, l2 = timeInterval
    ilim1 = indice (l1, T)
    ilim2 = indice (l2, T)

    for i in range(n):
        offset = np.mean(Xs[i][ilim1:ilim2])
        Xs[i] -= offset
        Yp = Ys[n-1] - Ys[i]
        if i!=n-4 and i!=n-1 :
            A, w, phi = curve_fit(SinModel.sinus, T[ilim1:ilim2], Xs[i][ilim1:ilim2], initial)[0]
            s = SinModel(A, w, phi)
            exec(f'p = Point({i}, None, Xs[i], s, np.mean(Yp))')
            exec('tour.points.append(p)')

    return tour



L = ['test lego tour A basique.mp4',
     'test lego tour B etage.mp4',
     'test lego tour C triangle.mp4',
     'test lego tour D triangle etage.mp4']

VIs = [[4.3, 6.7], [1.5, 4.8], [1, 3.3], [0.8, 3.6]]

Tours = []

for i in range(len(L)):

    assert len(L) == len(VIs)

    tour = analysis(L[i], VIs[i])

    Tours.append(tour)
    T = tour.time
    n = tour.markercount


    plt.figure('vérification de la validité des modèles - ' + L[i])
    plt.clf()

    for p in tour.points:
        plt.subplot(n//2, 2, p.id+1)
        plt.title('X'+str(p.id))
        A, w, phi = p.posMod.amplitude, p.posMod.pulsation, p.posMod.dephasage
        plt.plot(T, SinModel.sinus(T, A, w, phi), label='modèle', color='blue')
        plt.plot(T, p.posMes, label='mesure', color='red')
        # plt.legend()
        plt.grid()

    plt.show()


## affichage du résultat de l'étude

plt.figure('premiers résultats')
plt.clf()


plt.subplot(2, 2, 1)

for tour in Tours:

    Altitudes = [p.altitude for p in tour.points]
    Amplitudes = [abs(p.posMod.amplitude) for p in tour.points]

    plt.plot(Altitudes, Amplitudes, 'o-', label=tour.id)
    plt.xlabel('Altitude (en cm)')
    plt.ylabel('Amplitude (en cm)')
plt.grid()
plt.legend()


plt.subplot(2, 2, 2)

for tour in Tours:

    Altitudes = [p.altitude for p in tour.points]
    Dephasages = [p.posMod.dephasage-tour.points[-1].posMod.dephasage for p in tour.points]

    plt.plot(Altitudes, Dephasages, 'o-', label=tour.id)
    plt.xlabel('Altitude (en cm)')
    plt.ylabel('dephasage (en rad)')
plt.grid()
plt.legend()


plt.subplot(2, 2, 3)

for tour in Tours:

    Altitudes = [p.altitude for p in tour.points]
    Pulsations = [p.posMod.pulsation for p in tour.points]

    plt.plot(Altitudes, Pulsations, 'o-', label=tour.id)
    plt.xlabel('Altitude (en cm)')
    plt.ylabel('Pulsations (en rad/s)')
plt.ylim(32, 38)
plt.grid()
plt.legend()


plt.show()



