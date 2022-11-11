import numpy as np
import cv2 as cv2
import csv as csv
import pymediainfo as mi
import os as os             # intégré à python par défaut
import sys as sys           # intégré à python par défaut
import time as t            # intégré à python par défaut
import getpass as gp        # intégré à python par défaut
import shutil as sht        # intégré à python par défaut


stoplist = ['stop', 'quit', 'abandon', 'kill']
class Break (Exception):
    pass


class Settings:
    def __init__(self, video):
        sys.setrecursionlimit(100)  # permet de gérer la precision du système
        self.tol = 0.4              # est réglable lors de l'execution
        self.definition = 1         # est automatiquement réglé par le programme
        self.step = 1               # est automatiquement réglé par le programme

        # On définit la taille des indicateurs visuels / taille de l'image
        self.minsize = int(video.Framessize[1] / 50)
        self.maxdist = int(video.Framessize[1] / (0.25 * video.Framerate))
        self.bordure_size = int(video.Framessize[1] /  video.Framerate * 2)
        self.crosswidth = int(video.Framessize[1] / 500)
        self.rectanglewidth = int(video.Framessize[1] / 1250)

class SettingError (Exception):
    pass


class Video:
    def __init__(self, id):

        self.id = id

        self.Frames = None
        self.Framerate = None
        self.Framessize = None

        self.markerscolor = None
        self.orientation = None
        self.lenref = None

        self.scale = None
        self.markercount = None


        self.get_frames()
        self.get_framerate()
        self.get_framessize()

        self.markerscolor_input()
        self.orientation_input()
        self.ref_input()

    def get_frames(self):
        """
        Renvoie un dictionaire où les clés sont les numéros de frames
        et les valeurs sont les images (tableaux de type uint8).
        """
        frames = []
        cam = cv2.VideoCapture(paths['vidéoinput'])
        frame_number = 0
        print('\rRécupération de la vidéo en cours ...', end='')
        while True:
            ret, frame = cam.read()
            if ret:
                frames.append(Frame('frame.' + str(frame_number), frame))
                frame_number += 1
            else:
                break
        cam.release()
        cv2.destroyAllWindows()
        print('\rRécupération de la vidéo ------------------------------------------ OK')
        t.sleep(0.1)
        self.Frames = frames

    def get_framerate(self):
        """
        Renvoie le nombre de frames par secondes de la vidéo passée en entrée du
            script.
        """
        media_info = mi.MediaInfo.parse(paths['vidéoinput'])
        tracks = media_info.tracks
        for i in tracks:
            if i.track_type == 'Video':
                Framerate = float(i.frame_rate)
        self.Framerate = Framerate

    def get_framessize(self) -> tuple:
        """
        Renvoie un tuple de deux valeurs : la hauteur et largeur des frames de
        la video.
        """
        media_info = mi.MediaInfo.parse(paths['vidéoinput'])
        video_tracks = media_info.video_tracks[0]
        Framessize = [int(video_tracks.sampled_width), int(video_tracks.sampled_height)]
        self.Framessize = Framessize

    def markerscolor_input(self):
        global stoplist
        while True :
            c = input('\nCouleur des repères à étudier (1=bleu, 2=vert, 3=rouge) : ')
            if c in ['1', '2', '3']:
                c = int(c)-1
                self.markerscolor = c
                break
            elif c in stoplist :
                raise Break
            else:
                print('Vous devez avoir fait une erreur, veuillez rééssayer.')

    def orientation_input(self):
        global stoplist
        Framessize = self.Framessize
        while True:
            mode = input('La vidéo est en mode (1=landscape, 2=portrait) : ')
            if mode in ['1', '2']:
                if mode == '1':
                    height = min(Framessize)
                    width = max(Framessize)
                elif mode == '2':
                    height = max(Framessize)
                    width = min(Framessize)
                Framessize = (width, height)
                self.Framessize = Framessize
                self.orientation = int(mode)
                break
            elif mode in stoplist :
                raise Break
            else:
                print('Vous devez avoir fait une erreur, veuillez rééssayer.')

    def ref_input(self):
        global stoplist
        while True:
            l = input('Longueur entre les deux premiers repères(cm) : ')
            try :
                if l in stoplist:
                    raise Break
                else :
                    lenref = float(l)
                    self.lenref = lenref
                    break
            except ValueError :
                print('Vous devez avoir fait une erreur, veuillez rééssayer.')

class Frame:
    def __init__(self, id, array):
        self.id = id
        self.array = array
        self.identified_objects = []

class Object:
    def __init__(self, id):
        self.id = id
        self.status = 'in'
        self.positions = {}
        self.LastKnownPos = None


# main
def main():
    """
    """
    global video, settings # pour pouvoir accéder à ces données une fois le  traitement finit.

    print('\nInitialisation de la procédure')

    try :

        # On récupère la vidéo et ses caractéristiques
        create_dir('bac')
        video = Video(videoinput())
        delete_dir('bac')

        # On definit les réglages par défault
        settings = Settings(video)

        # On traite la première frame  pour vérifier que les reglages sont bons
        isOK = False
        while not isOK:
            calibration()
            if yn('Le traitement est-il bon ?'):
                isOK = True
            else:
                verif_settings()
                settings.definition, settings.step = 1, 1
                video.Frames[0].identified_objects = []

        # Une fois que tout est bon on traite la vidéo
        videotreatement()

        # On télécharge les données
        reboot(video)
        datadownload(video)

        if yn("Voulez vous télécharger les résultats de l'étude ?"):
            resultsdownload(video, settings.crosswidth)

        print('\nProcédure terminée')

    except (Break, KeyboardInterrupt):
        print('\n\nProcédure terminée')

    cleaner()
    return None

def cleaner():
    sys.setrecursionlimit(1000)
    for i in range(3):
        delete_dir(L_paths[i])
    return None



# Calibration fcts

def calibration():
    """
    À effectuer avant le traitement de l'ensemble de la vidéo pour vérifier le
        bon réglage de l'ensmeble des paramètres.
    """
    global video, settings

    print('\nTraitement en cours ...', end='')
    first = copy_im(video.Frames[0].array)

    try :
        detected = frametreatement(first, video.markerscolor)
        extremas = detected[0]
        settings.step = Pas(extremas, settings.definition)
        positions = position( rectifyer(detected[0], settings.minsize) )
        detScale(video, positions)
    except SettingError :
        print('\nIl y a un problème, veuillez vérifiez les réglages')
        verif_settings()
        settings.definition, settings.step = 1, 1
        calibration()
        return None



    video.markercount = 0
    for obj in positions :
        new_obj = Object('obj-'+str(video.markercount))
        new_obj.positions[video.Frames[0].id] = positions[obj]
        video.Frames[0].identified_objects.append(new_obj)
        video.markercount += 1

    print('\rTraitement -------------------------------------------------------- OK')

    images_names = []
    create_dir('calib')

    color_im = copy_im(first)
    images_names.append('color_im')
    fill_calibdir(color_im, 'color_im')

    NB_im = cv2.resize(detected[1], video.Framessize)
    images_names.append('NB_im')
    fill_calibdir(NB_im, 'NB_im')

    treated_NB = visu_detection(NB_im, detected[2])
    treated_NB = draw_rectangle_NB(treated_NB, extremas, settings.rectanglewidth)
    images_names.append('treated_NB')
    fill_calibdir(treated_NB, 'treated_NB')

    # treated_color = Add_pas(color_im, pas)
    treated_color = draw_cross_color(color_im, video.Frames[0], settings.crosswidth)
    treated_color = Add_scale(treated_color, video.scale,settings.crosswidth, settings.bordure_size, video.markerscolor)
    images_names.append('treated_color')
    fill_calibdir(treated_color, 'treated_color')

    print("\nAffichage du résultat (une fenêtre a dû s'ouvrir)", end='')
    calib_show(images_names)
    print('\rValidation du résultat -------------------------------------------- OK')
    t.sleep(0.1)
    sht.rmtree(paths['calib'])

    return None

def copy_im (image:np.array) -> np.array:
    '''
    copie l'image passée en argument de manière a défaire le lien entre les
        objets.
    '''
    L = len(image)
    l = len(image[0])
    newIm = []
    for y in range (L):
        newLine = []
        for x in range(l):
            newLine.append(image[y][x])
        newIm.append(newLine)
    return np.uint8(newIm)

def fill_calibdir(image:np.array, image_name:str):
    '''
    permet d'enregistrer l'image passée en argument dans le dossier de calibration
    '''
    cv2.imwrite(paths['calib'] + '/' + image_name + '.jpg', image)
    return None

def calib_show(images_names: list):
    for i in range(len(images_names)):
        name = 'Config Window - ' + images_names[i]
        cv2.imshow(name, cv2.imread(paths['calib'] + '/' + images_names[i] + '.jpg'))
        cv2.waitKey(0)
        cv2.destroyWindow(name)
        cv2.waitKey(1)
    return None



# Treatement tools

def videotreatement() -> None:
    """
    Permet le traitement de l'ensemble des frames qui constituent la vidéo ainsi
        que le suivi des objets
    """
    global video, settings
    frames = video.Frames
    markerscolor = video.markerscolor
    minsize, maxdist = settings.minsize, settings.maxdist
    bordure_size = settings.bordure_size



    Ti, T = t.time(), t.time()
    print()

    for i in range(1, len(frames)): # frame 0 deja traitée durant l'initialisation
        try :

            objects_extremums = frametreatement(frames[i].array, markerscolor)[0]
            objects_extremums = rectifyer(objects_extremums, minsize)
            positions = position(objects_extremums)

            object_tracker(video, i, positions, maxdist, bordure_size)

        except SettingError :
            print('problèmes dans les réglages')

        if t.time() - T >= 1 :
            progression = round((int(frames[i].id.split('.')[1]) / (len(frames) - 1)) * 100, 1)
            print('\rTraitement en cours :', str(progression), '% (' + waiting_time(i, len(frames), Ti) + ')', end='')
            T = t.time()
    t.sleep(1)
    print('\rTraitement de ' + video.id + ' ' + '-'*(67-(14+len(video.id))) + ' OK')
    t.sleep(1)
    return None

def object_tracker(video, i, positions, maxdist, bordure_size):
    frames = video.Frames

    for obj1 in positions :

        identified = False
        distances_list = {}
        x1, y1 = positions[obj1][0], positions[obj1][1]

        for obj2 in video.Frames[i-1].identified_objects:
            x2, y2 = obj2.positions[video.Frames[i-1].id][0], obj2.positions[video.Frames[i-1].id][1]
            d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            distances_list[obj2] = d

        if len(distances_list) != 0:
            min_key = min(distances_list, key=distances_list.get)
            distance = distances_list[min_key]
            if distance < maxdist:
                identified = True
                video.Frames[i].identified_objects.append(min_key)
                min_key.positions[video.Frames[i].id] = positions[obj1]

        if not identified :
            if in_bordure(bordure_size, positions[obj1]):
                new_obj = Object('obj-'+str(obj_compteur))
                new_obj.positions[video.frames[i].id] = [x1, y1]
                video.Frames[i].identified_objects.append(new_obj)
                video.markercount += 1

        for obj in video.Frames[i-1].identified_objects:
            if not obj in video.Frames[i].identified_objects:
                obj.status = 'out'
                obj.LastKnownPos = obj.positions[frames[i-1].id]

def in_bordure (bordure_size, pos):
    # Les objets apparaissant aux bordures de l'écran ne seront pas considérés comme des erreurs
    # mais comme des nouveaux objetsentrant dans le chant de la caméra.
    BandeGaucheHaut = [i for i in range(0, bordure_size + 1)]
    BandeBas = [i for i in range(video.Framessize[1] - bordure_size, video.Framessize[1] + 1)]
    BandeDroite = [i for i in range(video.Framessize[0] - bordure_size, video.Framessize[0] + 1)]

    x1, y1 = pos[0], pos[1]
    if x1 in BandeGaucheHaut or x1 in BandeDroite:
        return True
    if y1 in BandeGaucheHaut or y1 in BandeBas:
        return True
    else :
        return False

def waiting_time(i, N, Ti):
    d = t.time()-Ti
    d = round((N-i)*(d/i), 1)
    minutes = str(int(d//60))
    if int(minutes) < 10 :
        minutes = '0' + minutes
    secondes = str(int(d%60))
    if int(secondes) < 10 :
        secondes = '0' + secondes
    return minutes + 'min ' + secondes + 'sec'

def frametreatement(frame:np.array, c:int) -> tuple:
    """
    frame : image à traiter (tableau uint8).
    tol : seuil à partir duquel on détecte un objet.
    c : couleur des repères étudiés.
    minsize : taille minimum à partir de laquelle on détecte un objet.
    pas : distance entre chaque pixel que l'on va tester.

    Permet le traitement de la frame passée en argument.
    """
    global settings
    definition = settings.definition
    isOK = False
    while not isOK and definition <= 15:
        try:
            NB_im = prep(frame, settings.definition, settings.tol, c)
            extremas, borders = objects_identification(NB_im, settings.definition, settings.step)
            isOK = True
        except RecursionError:
            print('\rDéfinition trop élevée, tentative avec une défintion plus faible', end='')
            settings.definition += 1
    if isOK:
        return extremas, NB_im, borders
    else:
        raise SettingError



# Frame manipulation tools

def objects_identification(image:np.array, definition:int, pas:int) -> dict:
    """
    image : image à traiter.
    defintion : facteur avec lequel on réduit notre frame pour simplifier le
        traitement.
    pas :  distance entre chaque pixel que l'on va tester.

    Regroupe tout les objets de l'image dans un dictionnaire.
    image : image en N&B sous la forme d'un array de 0 et 255.
    """
    h = len(image)
    w = len(image[0])
    extremas = {}
    borders = {}
    n = 0

    for j in range(int(h/pas)):
        for i in range(int(w/pas)):
            if image[j*pas][i*pas] == 255:

                element_in = False
                for obj in extremas :
                    if  extremas[obj][1] <= j*pas <= extremas[obj][3] and extremas[obj][0] <= i*pas <= extremas[obj][2]:
                        element_in = True

                if not element_in :
                    extremas[n], borders[n] = discovery(image, [i*pas, j*pas])
                    n += 1

    for obj in extremas:
        xmin, ymin = extremas[obj][0], extremas[obj][1]
        xmax, ymax = extremas[obj][2], extremas[obj][3]
        extremas[obj] = [xmin*definition, ymin*definition, xmax*definition, ymax*definition]
        for i in range (len(borders[obj])):
            x, y = borders[obj][i][0], borders[obj][i][1]
            borders[obj][i] = [x*definition, y*definition]

    return extremas, borders

def discovery(image:np.array, depart:list) -> list:
    '''
    Permet l'initialisation pour la fonction recursive visiter.
    '''
    global at_border
    object = [depart]
    init_extr = [depart[0], depart[1], depart[0], depart[1]]
    at_border = False

    return detection(image, depart, object, init_extr)

def detection(image, depart: list, object: list, extr: list) -> list:
    """
    Regroupe tous les pixels appartenant a un même objets (forme blanche ici)
        sous la forme d'une liste.
    image : image en N&B.
    depart : pixel duquel on va partir pour 'explorer' notre objet,
        sous la forme [j,i].
    objet : liste contenant tout les pixels appartenants au même objet.
    """
    if depart not in object:            # but: récupérer un encadrement de objet
        object.append(depart)
        if depart[0] < extr[0]:
            extr[0] = depart[0]
        elif depart[1] < extr[1]:
            extr[1] = depart[1]
        if depart[0] > extr[2]:
            extr[2] = depart[0]
        elif depart[1] > extr[3]:
            extr[3] = depart[1]

    for pixel in get_neighbours(image, depart):
        if pixel not in object:
            detection(image, pixel, object, extr)

    return extr, object

def get_neighbours(image:np.array, pixel:list) -> list:
    """
    Renvoie la liste des voisins du pixel 'pixel' à étudier dans le cadre de la
        recherche d'objet.
    image : image en N&B.
    pixel : sous la forme [j,i].
    """
    global at_border
    x, y = pixel[0], pixel[1]
    h = len(image)
    w = len(image[0])
    view = 2

    neighbours_coordinates = []
    for i in range (-view, view+1):
        for j in range (-view, view+1):
            if j != i :
                neighbours_coordinates.append([(x+i)%w, (y+j)%h])

    is_border = False
    outsiders = []
    for n in neighbours_coordinates :
        if image[n[1]][n[0]] == 0 :
            is_border = True
            at_border = True
            outsiders.append(n)

    L_neighbours = []
    if not is_border and not at_border:
        L_neighbours.append([pixel[0]+1, pixel[1]])
    if is_border :
        for n in neighbours_coordinates :
            if n not in outsiders :
                for o in outsiders :
                    if abs(n[0]-o[0]) <= 1 and abs(n[1]-o[1]) <= 1 :
                        L_neighbours.append(n)

    return L_neighbours

def position(extremas:dict) -> list:
    """
    Récupère la position d'un objet à partir des extremas.
    Renvoie un dictionnaire où les clefs sont les noms des ifférents objets
        détectés sur la frame étudiée et les valeurs sont les coordonées
        du 'centre' de l'objet.
    """
    position = {}
    for obj in extremas:
        x = (extremas[obj][0] + extremas[obj][2]) / 2
        y = (extremas[obj][1] + extremas[obj][3]) / 2
        position[obj] = [x, y]
    return position

def rectifyer(extremas:dict, minsize:int) -> dict:
    """
    Rectifie quelques erreurs.
    """
    # On supprime les objets trop petits, probablement issus d'erreurs.
    problematic_objects = []
    for obj in extremas:
        if extremas[obj][2] - extremas[obj][0] < minsize or extremas[obj][3] - extremas[obj][1] < minsize:
            problematic_objects.append(obj)
    for obj in problematic_objects:
        del extremas[obj]
    return extremas


def rate_rgb(pixel:list, c:int) -> float:
    """
    pixel : élement de l'image d'origine sous la forme [r, g, b].
    c = 0(rouge), 1(vert) ou 2(bleu).

    Calcul le poids relatif de la composante c du pixel pixel parmis les
        composantes rgb qui le définissent.
    """
    assert c in [0, 1, 2]
    return int(pixel[c]) / (int(pixel[0]) + int(pixel[1]) + int(pixel[2]) + 1)

def prep(image:np.array, definition:int, tol:float, c:int) -> np.array:
    """
    image : image de depart.
    Definition : l'image finale contiendra 1/definition² pixels de l'image
        initiale.

    Renvoie une image en noir et blanc
    """
    assert 0 < definition
    assert type(definition) == int
    simplified_im = []
    h = len(image)
    w = len(image[0])
    for i in range(int(h / definition)):
        line = []
        for j in range(int(w / definition)):
            pixel = image[i * definition][j * definition]
            if rate_rgb(pixel, c) < tol:
                line.append(0)
            else:
                line.append(255)
        simplified_im.append(line)
    return np.uint8(simplified_im)

def Pas (extr:dict, definition:int):
    '''
    extre : {0: [xmin, ymin, xmax, ymax], 1: ... }
        dictionaire où chaque clef correspond à un objet,
        la valeure qui lui est associée est la liste des 4 coordonées
        extremales entourant l'objet.
    '''
    L = list(extr.keys())
    if len(L) == 0:
        raise SettingError
    min = extr[L[0]][2]-extr[L[0]][0]
    for el in extr :
        if extr[el][2]-extr[el][0] < min :
            min = extr[el][2]-extr[el][0]
        if extr[el][3]-extr[el][1] < min :
            min = extr[el][3]-extr[el][1]
    return int(min/(definition* 3)) + 1 # On multiplie par 3 pour s'assurer de ne manquer aucun repère.



# indicateurs visiuels sur la vidéo

def draw_rectangle_NB(image:np.array, extremas:dict, rectanglewidth:int) -> np.array:
    L = len(image)
    l = len(image[0])
    marge = 4
    for key in extremas:
        xmin, ymin = int(extremas[key][0])-marge, int(extremas[key][1])-marge
        xmax, ymax = int(extremas[key][2])+marge, int(extremas[key][3])+marge
        for i in range(xmin - rectanglewidth, xmax + rectanglewidth + 1):
            for n in range(rectanglewidth + 1):
                if 0 <= i < l and 0 <= ymin-n < L and 0 <= ymin+n < L :
                    image[(ymin - n) % L][i % l], image[(ymax + n) % L][i % l] = 255, 255
        for j in range(ymin - rectanglewidth, ymax + rectanglewidth + 1):
            for n in range(rectanglewidth + 1):
                if 0 <= xmin-n < l and 0 <= xmin+n < l and 0 <= j < L :
                    image[j % L][(xmin - n) % l], image[j % L][(xmax + n) % l] = 255, 255
    return np.uint8(image)

def draw_cross_color(image:np.array, frame:Frame, crosswidth:int) -> np.array:
    L = len(image)
    l = len(image[0])
    for obj in frame.identified_objects :
        x = int(obj.positions[frame.id][0])
        y = int(obj.positions[frame.id][1])
        for i in range(x - crosswidth * 10, x + crosswidth * 10 + 1):
            for n in range(y - int(crosswidth / 2), y + int(crosswidth / 2) + 1):
                if 0<=i<l and 0<=n<L :
                    image[n][i] = [0, 255, 0]
        for j in range(y - crosswidth * 10, y + crosswidth * 10 + 1):
            for n in range(x - int(crosswidth / 2), x + int(crosswidth / 2) + 1):
                if 0 <= n < l and 0 <= j < L :
                    image[j][n] = [0, 255, 0]
    return np.uint8(image)

def Add_pas (image:np.array, pas:int) -> np.array:
    if pas >= 2 :
        for j in range (int(len(image)/pas)):
            for i in range (int(len(image[j])/pas)):
                image[j*pas][i*pas] = [0, 0, 0]
    return np.uint8(image)

def Add_scale(image:np.array, scale:float, crosswidth:int, bordure_size:int, c:int) -> np.array:
    L = len(image)
    l = len(image[0])
    color = [0, 0, 0]
    color[video.markerscolor] = 255
    for i in range(int(1/scale)):
        for j in range(crosswidth):
            image[(j+L-bordure_size+10) % L][(bordure_size+i) % l] = color
    cv2.putText(image, '1cm', (bordure_size, L-bordure_size-3), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
    return np.uint8(image)

def visu_detection (image:np.array, borders:list) -> np.array:
    global definition
    L = len(image)
    l = len(image[0])
    for j in range(L) :
        for i in range(l):
            if image[j][i] == 255:
                image[j][i] = 100
    for obj in borders:
        for pixel in borders[obj] :
            for i in range (-1, 2):
                for j in range (-1, 2):
                    if 0 <= pixel[1] < L and 0 <= pixel[0] < l :
                        image[pixel[1]+j][pixel[0]+i] = 255
    return np.uint8(image)



# path gestion

user = gp.getuser()
paths = {}
L_paths = ['bac', 'calib', 'video storage', 'data']
paths[L_paths[0]] = '/Users/' + user + '/Desktop/bac'
paths[L_paths[1]] = '/Users/' + user + '/Desktop/.##calibdir##'
paths[L_paths[2]] = '/Users/' + user + '/Desktop/.##temporary storage##'
paths[L_paths[3]] = '/Users/' + user + '/Desktop/data'

if os.name == 'nt':
    for el in L_paths:
        paths[el] = 'C:'+paths[el]

def add_subdata_dirs(video:str) -> None:
    '''
    video : nom de la video passée en entrée du script.

    Permet d'ajouter les dossier propre à la vidéo dans le dossier data (où les
        données sont stockées).
    '''
    paths['csv'] = paths['data'] + '/' + video + '/csv'
    paths['vidéodl'] = paths['data'] + '/' + video + '/vidéo'
    paths['frames'] = paths['data'] + '/' + video + '/frames'
    paths['treated frames'] = paths['frames'] + '/treated'
    paths['non treated frames'] = paths['frames'] + '/non treated'
    return None

def create_dir(dir:str) -> None:
    '''
    dir : nom du dossier à créer.

    Permet de créer le dossier dont le nom est passé en argument à l'endroit
        qui lui est prédestiné dans paths.
    '''
    p = paths[dir]
    if not os.path.exists(p):
        os.makedirs(p)
    return None

def delete_dir(dir:str) -> None:
    '''
    dir : nom du dossier à supprimer.

    Permet de supprimer le dossier dont le nom est passé en argument à l'endroit
        qui lui est prédestiné dans paths.
    '''
    p = paths[dir]
    if os.path.exists(p) :
        sht.rmtree(p)
    return None



# fonctions permettant l'IHM

def videoinput() :
    isempty = True
    print('\nPlacez la vidéo (.mp4) à étudier dans le bac sur votre bureau.', end='')
    while isempty:
        if len(os.listdir(paths['bac'])) != 0:
            isempty = False
        t.sleep(0.5)
    bac = os.listdir(paths['bac'])
    if len(bac) == 1 and bac[0].split('.')[1] == 'mp4':
        video = bac[0].split('.')[0]
        paths['vidéoinput'] = paths['bac'] + '/' + video + '.mp4'
        create_dir('video storage')
        sht.copy2(paths['vidéoinput'], paths['video storage'])
        return  video
    elif len(bac) == 1 and bac[0].split('.')[1] != 'mp4':
        print('Veuillez fournir une vidéo au format mp4')
        delete_dir('bac')
        videoinput()
    elif len(bac) > 1:
        print("Veuillez ne placer qu'un document dans le bac")
        delete_dir('bac')
        videoinput()

def verif_settings ():
    global video, settings
    while True :
        print('\n1 couleur des repères :', ['bleue', 'verte', 'rouge'][video.markerscolor])
        print('2 orientation de la vidéo :', ['landscape', 'portrait'][video.orientation-1])
        print('3 longueur de référence : ', video.lenref)
        print('4 tolérance : ', settings.tol, 'cm')
        which = input('quel réglage vous semble-t-il éroné (0=aucun, 1, 2, 3, 4) ? ')
        if which in ['0', '1', '2', '3', '4', 'pres']:
            if which == '0':
                pass
            elif which == '1':
                video.markerscolor_input()
            elif which == '2':
                video.orientation_input()
            elif which == '3':
                video.ref_input()
            elif which == '4':
                settings.tol += float(input('\nTolérance actuelle : ' + str(settings.tol) + ', implémenter de : '))
            elif which == 'pres':
                sys.setrecursionlimit(int(input('setrecursionlimit : ')))
            return None
        elif which in stoplist :
            raise Break
        else:
            print ('vous devez avoir fait une erreur, veuillez réessayer')

def yn(question):
    assert type(question) == str
    while True:
        yn = input('\n' + question + ' [y]/n : ')
        if yn in ['y', '', 'n']:
            if yn == 'y' or yn == '':
                return True
            elif yn == 'n':
                return False
        elif yn in stoplist :
            raise Break
        else:
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def detScale (video:Video, positions:dict) -> float:
    '''
    positions : dictionaire contenant les positions de chaque repère sur
        chacune des frames.
    lenref : longeur de reférance sur laquelle on s'appuie pour définir
        l'échelle

    Renvoie l'échelle de la vidéo en cm par nb de pixel
    '''
    lenref = video.lenref
    if len(positions) >= 2 :
        a = list(positions.keys())[0]
        b = list(positions.keys())[1]
        apos, bpos = positions[a], positions[b]
        xa , ya , xb, yb = apos[0], apos[1], bpos[0], bpos[1]
        scale = lenref / ( ( (xa-xb)**2 + (ya-yb)**2 )**(1/2) )
    else :
        scale = 1
    video.scale = scale
    return scale


# Récupération des résultats du traitement

def resultsdownload(video, crosswidth):
    videodownload(video)
    create_video(video, crosswidth)
    # framesdownload(video, crosswidth)
    return None

def reboot(video):
    add_subdata_dirs(video.id)
    delete_dir('csv')
    delete_dir('frames')
    delete_dir('vidéodl')
    add_subdata_dirs(video.id)
    return None

def videodownload(video):
    create_dir('vidéodl')
    source = paths['video storage']  + '/' + video.id + '.mp4'
    destination = paths['vidéodl'] + '/vidéo' + '.mp4'
    sht.copy2(source, destination)
    sht.rmtree(paths['video storage'])
    return None

def datadownload(video):
    create_dir('csv')
    print('\nSauvegarde de la data en cours ...', end='')
    nom_colonnes = ['frame', 'time']
    objects = []
    frames = video.Frames
    for frame in frames:
        for obj in frame.identified_objects:
            if obj not in objects:
                objects.append(obj)
                nom_colonnes += ['X' + obj.id, 'Y' + obj.id]
    dos = open(paths['csv'] + '/positions objets.csv', 'w')
    array = csv.DictWriter(dos, fieldnames=nom_colonnes)
    array.writeheader()
    for frame in frames:
        dico = {'frame': frame.id, 'time': round(int(frame.id.split('.')[1]) / video.Framerate, 3)}
        for obj in frame.identified_objects:
            dico['X' + obj.id] = video.scale * obj.positions[frame.id][0]
            dico['Y' + obj.id] = video.scale * obj.positions[frame.id][1]
        array.writerow(dico)
    dos.close()
    t.sleep(1)
    print('\rSauvegarde de la data --------------------------------------------- OK')
    return None

def framesdownload(video, crosswidth):
    create_dir('non treated frames')
    create_dir('treated frames')
    print('\nSauvegarde des frames en cours ...', end='')
    for frame in video.frames:
        name = paths['non treated frames'] + '/frame' + str(int(frame.id.split('.')[1])) + '.jpg'
        cv2.imwrite(name, frame.array)
        name = paths['treated frames'] + '/frame' + str(int(frame.id.split('.')[1])) + '.jpg'
        im = draw_cross_color(frame.array, frame.identified_objects, crosswidth)
        cv2.imwrite(name, im)
    print('\rSauvegarde des frames --------------------------------------------- OK')
    return None

def create_video(video, crosswidth):
    global pas
    out = cv2.VideoWriter(paths['vidéodl'] + '/vidéo traitée' + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video.Framerate, video.Framessize)
    print('\nSauvegarde de la vidéo en cours ...', end='')
    for frame in video.Frames:
        img = draw_cross_color(frame.array, frame, crosswidth)
        # img = Add_pas(img, pas)
        out.write(img)
    print('\rSauvegarde de la vidéo -------------------------------------------- OK')
    return None

main()