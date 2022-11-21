import numpy as np
import cv2
import csv
import pymediainfo as mi
import os                   # intégré à python par défaut
import sys                  # intégré à python par défaut
import time as t            # intégré à python par défaut
import getpass as gp        # intégré à python par défaut
import shutil as sht        # intégré à python par défaut
import inspect


stoplist = ['stop', 'quit', 'abandon', 'kill']
class Break (Exception):
    pass


class Settings:
    def __init__(self, video):

        self.precision = 100       # permet de gérer la precision du système
        self.tol = 0.4             # est réglable lors de l'execution
        self.maxdef = 15           # abaissement de la definition maximal
        self.definition = 1        # est automatiquement réglé par le programme
        self.step = 1              # est automatiquement réglé par le programme
        sys.setrecursionlimit(self.precision)

        # On définit la taille des indicateurs visuels / taille de l'image
        self.minsize = int(video.Framessize[1] / 170)
        # self.maxdist = int(video.Framessize[1] / (0.25 * video.Framerate) * 5)
        # self.bordure_size = int(video.Framessize[0] /  video.Framerate * 2)
        self.maxdist = int(video.Framessize[1] / video.Framerate * 10)
        self.bordure_size = 0
        self.crosswidth = int(video.Framessize[1] / 500)
        self.rectanglewidth = int(video.Framessize[1] / 1250)

class SettingError (Exception):
    pass


class Video:

    def __init__(self):

        self.id = None
        self.name = None
        self.videoinput()

        self.Frames = self.get_frames()
        self.Framerate = self.get_framerate()
        self.Framessize = self.get_framessize()

        self.markerscolor = None
        self.orientation = None
        self.lenref = None
        self.markerscolor_input()
        self.orientation_input()
        self.ref_input()

        self.scale = None
        self.markercount = None

    def videoinput(self) :
        create_dir('bac')
        isempty = True
        print('Placez la vidéo à étudier dans le bac sur votre bureau.', end='')
        while isempty:
            if len(os.listdir(paths['bac'])) != 0:
                isempty = False
            t.sleep(0.5)
        bac = os.listdir(paths['bac'])
        ext = bac[0].split('.')[1]
        if len(bac) == 1 and (ext == 'mp4' or ext == 'mov'):
            video = bac[0]
            paths['vidéoinput'] = paths['bac'] + '/' + video
            create_dir('video storage')
            sht.copy2(paths['vidéoinput'], paths['video storage'])
            self.id = str(video)
            self.name = ''.join( tuple( video.split('́') ) )
            delete_dir('bac')
        elif len(bac) == 1 and ext != 'mp4' and ext != 'mov' :
            print('\rVeuillez fournir une vidéo au format mp4', end='')
            delete_dir('bac')
            self.videoinput()
        elif len(bac) > 1:
            print("\rVeuillez ne placer qu'un document dans le bac", end='')
            delete_dir('bac')
            self.videoinput()

    def get_frames(self):
        """
        Renvoie une listes contenatnt l'ensembles des frames (tableaux de type
        uint8) dans le même ordre que dans la vidéo étudiée.
        """
        frames = []
        cam = cv2.VideoCapture(paths['video storage'] + '/' + self.id)
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
        print('\rRécupération de la vidéo ------------------------------------------ OK', end='\n\n')
        t.sleep(0.1)
        return frames

    def get_framerate(self):
        """
        Renvoie le nombre de frames par secondes de la vidéo passée en entrée du
        script.
        """
        media_info = mi.MediaInfo.parse(paths['video storage'] + '/' + self.id)
        tracks = media_info.tracks
        for i in tracks:
            if i.track_type == 'Video':
                framerate = float(i.frame_rate)
        return framerate

    def get_framessize(self) -> tuple:
        """
        Renvoie un tuple de deux valeurs : la hauteur et largeur des frames de
        la video.
        """
        media_info=mi.MediaInfo.parse(paths['video storage']+'/'+self.id)
        video_tracks = media_info.video_tracks[0]
        w = int(video_tracks.sampled_width)
        h = int(video_tracks.sampled_height)
        framessize = (w, h)
        return framessize

    def markerscolor_input(self) -> None:
        """
        Récupère au près de l'utilisateur la couleur des repères placés sur
        l'objet étudiée sur la vidéo et assigne cette valeur à l'attribut
        markerscolor de la vidéo.
        """
        global stoplist
        while True :
            c = input('Couleur des repères à étudier (1=bleu, 2=vert, 3=rouge) : ')
            if c in ['1', '2', '3']:
                c = int(c)-1
                self.markerscolor = c
                return None
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
        """
        Récupère au près de l'utilisateur la distances séparant les deux
        premiers repères placés sur l'objet étudiée sur la vidéo et assigne
        cette valeur à l'attribut lenref de la vidéo.
        """
        global stoplist
        while True:
            l = input('Longueur entre les deux premiers repères(cm) : ')
            try :
                if l in stoplist:
                    raise Break
                else :
                    lenref = float(l)
                    self.lenref = lenref
                    return None
            except ValueError :
                print('Vous devez avoir fait une erreur, veuillez rééssayer.')

class Frame:
    def __init__(self, id, array):
        self.id = id
        self.array = array
        self.AreasOfInterest = []
        self.identifiedObjects = []

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
    global video, settings # pour pouvoir accéder à ces données une fois le
                           # traitement finit.

    print('Initialisation de la procédure', end='\n\n')

    try :

        # On récupère la vidéo et ses caractéristiques
        video = Video()

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
                video.Frames[0].identifiedObjects = []

        # Une fois que tout est bon on traite la vidéo
        videotreatement()

        # On télécharge les données
        reboot()
        datadownload()

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

    print()
    print('Traitement en cours ...', end='')
    first = copy_im(video.Frames[0].array)

    try :

        detected = frametreatement(first)
        extremas = rectifyer(detected[0], settings.minsize)
        positions = position(extremas)

        settings.step = Pas(extremas, settings.definition)
        detScale(video.lenref, positions)

    except SettingError :
        print('Il y a un problème, veuillez vérifiez les réglages', end='\n' )
        verif_settings()
        settings.definition, settings.step = 1, 1
        video.Frames[0].identifiedObjects = []
        calibration()
        return None

    video.markercount = 0
    for obj in positions :
        new_obj = Object('obj-'+str(video.markercount))
        new_obj.positions[video.Frames[0].id] = positions[obj]
        video.Frames[0].identifiedObjects.append(new_obj)
        video.markercount += 1

    print('\rTraitement -------------------------------------------------------- OK', end='\n\n')
    t.sleep(0.1)
    print('Création des visuels en cours ...', end='')
    images_names = []
    create_dir('calib')

    color_im = copy_im(first)
    images_names.append('color_im')
    fill_calibdir(color_im, 'color_im')

    reduced = reducer(color_im, settings.definition)
    NB_im = visu_reduced(reduced)
    NB_im = cv2.resize(NB_im, video.Framessize)
    images_names.append('NB_im')
    fill_calibdir(NB_im, 'NB_im')

    treated_NB = visu_detection(NB_im, detected[1])
    treated_NB = draw_rectangle_NB(treated_NB, extremas, settings.rectanglewidth)
    images_names.append('treated_NB')
    fill_calibdir(treated_NB, 'treated_NB')

    # treated_color = Add_pas(color_im, pas)
    treated_color = draw_cross_color(color_im, video.Frames[0], settings.crosswidth)
    treated_color = Add_scale(treated_color, video.scale, settings.crosswidth, video.markerscolor)
    images_names.append('treated_color')
    fill_calibdir(treated_color, 'treated_color')

    print("\rAffichage du résultat (une fenêtre a dû s'ouvrir)", end='')
    calib_show(images_names)
    print('\rValidation du résultat -------------------------------------------- OK', end='\n')
    t.sleep(0.1)
    sht.rmtree(paths['calib'])

    return None


def fill_calibdir(image:np.array, image_name:str):
    '''
    Permet d'enregistrer l'image passée en argument dans le dossier de
    calibration
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
    minsize, maxdist = settings.minsize, settings.maxdist
    bordure_size = settings.bordure_size

    Ti, T = t.time(), t.time()
    print()

    for i in range(1, len(frames)): # frame 0 deja traitée durant l'initialisation
        try :

            markers_extremums = frametreatement(frames[i].array)[0]
            markers_extremums = rectifyer(markers_extremums, minsize)
            positions = position(markers_extremums)

            object_tracker(video, i, positions, maxdist, bordure_size)

        except SettingError :
            print('\rproblèmes dans les réglages')

        if t.time() - T >= 1 :
            progr = (int(frames[i].id.split('.')[1]) / (len(frames) - 1)) * 100
            progr = str(round(progr))
            tleft = waiting_time(i, len(frames), Ti)
            print('\rTraitement en cours : ' +progr+ ' % (' +tleft+ ')', end='')
            T = t.time()

    t.sleep(0.1)
    print('Traitement de ' + video.name + ' ' + '-'*( 67-15-len(video.name) ) + ' OK', end='\n\n')

    return None

def object_tracker(video, i, positions, maxdist, bordure_size):
    frames = video.Frames

    for obj1 in positions :

        identified = False
        distances_list = {}
        x1, y1 = positions[obj1][0], positions[obj1][1]

        for obj2 in frames[i-1].identifiedObjects:
            x2, y2 = obj2.positions[frames[i-1].id][0], obj2.positions[frames[i-1].id][1]
            d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            distances_list[obj2] = d

        if len(distances_list) != 0:
            min_key = min(distances_list, key=distances_list.get)
            distance = distances_list[min_key]
            if distance < maxdist:
                identified = True
                video.Frames[i].identifiedObjects.append(min_key)
                min_key.positions[video.Frames[i].id] = positions[obj1]

        if not identified :
            if in_bordure(bordure_size, positions[obj1]):
                new_obj = Object('obj-'+str(video.markercount))
                new_obj.positions[video.Frames[i].id] = [x1, y1]
                video.Frames[i].identifiedObjects.append(new_obj)
                video.markercount += 1

        for obj in video.Frames[i-1].identifiedObjects:
            if not obj in video.Frames[i].identifiedObjects:
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


def frametreatement(frame:np.array) -> tuple:
    """
    frame : image à traiter (tableau uint8).
    tol : seuil à partir duquel on détecte un objet.
    c : couleur des repères étudiés.
    minsize : taille minimum à partir de laquelle on détecte un objet.
    pas : distance entre chaque pixel que l'on va tester.

    Permet le traitement de la frame passée en argument.
    """
    global settings
    isOK = False
    while not isOK and settings.definition <= settings.maxdef :
        try:
            image = reducer(frame, settings.definition)
            extremas, borders = objects_identification(image)
            isOK = True
        except RecursionError:
            print('\rDéfinition trop élevée, tentative avec une défintion plus faible', end='')
            t.sleep(0.1)
            settings.definition += 1

    if isOK:
        definition = settings.definition
        for obj in extremas:
            xmin, ymin = extremas[obj][0]*definition, extremas[obj][1]*definition
            xmax, ymax = extremas[obj][2]*definition, extremas[obj][3]*definition
            extremas[obj] = [xmin, ymin, xmax, ymax]
            for i in range (len(borders[obj])):
                x, y = borders[obj][i][0]*definition, borders[obj][i][1]*definition
                borders[obj][i] = [x, y]
        return extremas, borders
    else:
        raise SettingError



# Frame manipulation tools

def objects_identification(image:np.array) -> tuple :
    """
    image : frame à traiter.
    """

    global video, settings
    global at_borders
    pas, definition, tol = settings.step, settings.definition, settings.tol
    markerscolor = video.markerscolor
    h = len(image)
    w = len(image[0])
    extremas = {}
    borders = {}
    n = 0


    for j in range(0, h, pas):
        for i in range(0, w, pas):

            element_in = False
            for obj in extremas :
                HorizontalAlignement = extremas[obj][1] <= j <= extremas[obj][3]
                VerticalAlignement = extremas[obj][0] <= i <= extremas[obj][2]
                if VerticalAlignement and HorizontalAlignement :
                    element_in = True

            if not element_in and rate_rgb(image[j][i], markerscolor) > tol :
                depart = [i, j]
                object = [depart]
                init_extr = [depart[0], depart[1], depart[0], depart[1]]
                at_border = False
                extremas[n], borders[n] = detection(image, depart, object, init_extr)
                n += 1

    return extremas, borders


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
    global video, settings
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
        if rate_rgb(image[n[1]][n[0]], video.markerscolor) < settings.tol :
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
    tosmall_objects = []

    for obj in extremas:
        if extremas[obj][2] - extremas[obj][0] < minsize :
            tosmall_objects.append(obj)
        elif extremas[obj][3] - extremas[obj][1] < minsize :
            tosmall_objects.append(obj)

    for obj in tosmall_objects:
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
    return int(pixel[c]) / (int(pixel[0]) + int(pixel[1]) + int(pixel[2]) + 0.1)


def reducer(image:np.array, definition:int) -> np.array:
    """
    image : image de depart.
    Definition : l'image finale contiendra 1/definition² pixels de l'image
    initiale.
    """
    simplified_im = []
    h = len(image)
    w = len(image[0])
    for i in range(0, h, definition):
        line = []
        for j in range(0, w, definition):
            line.append(image[i][j])
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
    min = extr[L[0]][2] - extr[L[0]][0]
    for el in extr :
        if extr[el][2] - extr[el][0] < min :
            min = extr[el][2] - extr[el][0]
        if extr[el][3] - extr[el][1] < min :
            min = extr[el][3] - extr[el][1]
    return int(min/(definition * 4)) # On multiplie par 3 pour s'assurer de ne manquer aucun repère.


def detScale (lenref:float, positions:dict) -> float:
    '''
    positions : dictionaire contenant les positions de chaque repère sur
        chacune des frames.
    lenref : longeur de reférance sur laquelle on s'appuie pour définir
        l'échelle

    Renvoie l'échelle de la vidéo en cm par nb de pixel
    '''
    if len(positions) >= 2 :
        a = list(positions.keys())[0]
        b = list(positions.keys())[1]
        apos, bpos = positions[a], positions[b]
        xa , ya , xb, yb = apos[0], apos[1], bpos[0], bpos[1]
        scale = lenref / ( ( (xa-xb)**2 + (ya-yb)**2 )**(1/2) )
    else :
        scale = 1
    video.scale = scale
    return None




# indicateurs visiuels sur la vidéo

def copy_im (image:np.array) -> np.array:
    '''
    Copie l'image passée en argument de manière a défaire le lien entre les
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

def draw_cross_color(image:np.array, frame:Frame, crosswidth:int) -> np.array:
    L = len(image)
    l = len(image[0])
    for obj in frame.identifiedObjects :
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

def visu_reduced(image:np.array) -> np.array :
    global video, settings
    h = len(image)
    w = len(image[0])
    newIm = []
    for j in range(h):
        newLine = []
        for i in range(w):
            if rate_rgb(image[j][i], video.markerscolor) > settings.tol:
                newLine.append(255)
            else :
                newLine.append(0)
        newIm.append(newLine)
    return np.uint8(newIm)

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

def Add_pas (image:np.array, pas:int) -> np.array:
    if pas >= 2 :
        for j in range (int(len(image)/pas)):
            for i in range (int(len(image[j])/pas)):
                image[j*pas][i*pas] = [0, 0, 0]
    return np.uint8(image)

def Add_scale(image:np.array, scale:float, crosswidth:int, c:int) -> np.array:
    L = len(image)
    l = len(image[0])
    color = [0, 0, 0]
    color[video.markerscolor] = 255
    for i in range(int(1/scale)):
        for j in range(crosswidth):
            image[j+L-int( L/20 )][i + int( l/10 )] = color
    cv2.putText(image, '1cm', (int(l/10) , L-int(L/20 + L/100)), cv2.FONT_HERSHEY_SIMPLEX, int(l/1000), color)
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
                    if 0 <= pixel[1] < L-j and 0 <= pixel[0] < l-i :
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

def verif_settings ():
    global video, settings
    while True :
        print('\n1 couleur des repères :', ['bleue', 'verte', 'rouge'][video.markerscolor])
        print('2 orientation de la vidéo :', ['landscape', 'portrait'][video.orientation-1])
        print('3 longueur de référence : ', video.lenref, 'cm')
        print('4 tolérance : ', settings.tol)
        which = input('quel est le réglage qui vous semble éroné (0=aucun, 1, 2, 3) ? ')
        if which in ['0', '1', '2', '3', 'pres']:
            if which == '0':
                pass
            elif which == '1':
                print()
                video.markerscolor_input()
            elif which == '2':
                print()
                video.orientation_input()
            elif which == '3':
                print()
                video.ref_input()
            elif which == '4':
                print()
                settings.tol += float(input('Tolérance actuelle : ' + str(settings.tol) + ', implémenter de : '))
                settings.tol = round(settings.tol, 3)
            elif which == 'pres':
                print()
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


# Récupération des résultats du traitement

def resultsdownload(video, crosswidth):
    videodownload(video)
    create_video(video, crosswidth)
    # framesdownload(video, crosswidth)
    return None

def reboot():
    global video
    add_subdata_dirs(video.id)
    delete_dir('csv')
    delete_dir('frames')
    delete_dir('vidéodl')
    add_subdata_dirs(video.id)
    return None

def videodownload(video):
    create_dir('vidéodl')
    source = paths['video storage']  + '/' + video.id
    destination = paths['vidéodl'] + '/vidéo' + '.mp4'
    sht.copy2(source, destination)
    sht.rmtree(paths['video storage'])
    return None

def datadownload():
    global video
    create_dir('csv')
    print('Sauvegarde de la data en cours ...', end='')
    nom_colonnes = ['frame', 'time']
    objects = []
    frames = video.Frames
    for frame in frames:
        for obj in frame.identifiedObjects:
            if obj not in objects:
                objects.append(obj)
                nom_colonnes += ['X' + obj.id, 'Y' + obj.id]
    dos = open(paths['csv'] + '/positions objets.csv', 'w')
    array = csv.DictWriter(dos, fieldnames=nom_colonnes)
    array.writeheader()
    for frame in frames:
        dico = {'frame': frame.id, 'time': round(int(frame.id.split('.')[1]) / video.Framerate, 3)}
        for obj in frame.identifiedObjects:
            dico['X' + obj.id] = video.scale * obj.positions[frame.id][0]
            dico['Y' + obj.id] = video.scale * obj.positions[frame.id][1]
        array.writerow(dico)
    dos.close()
    t.sleep(1)

    settingsdownload()

    print('\rSauvegarde de la data --------------------------------------------- OK')
    return None

def settingsdownload():
    global settings, video
    doc = open(paths['csv'] + '/settings.csv', 'w')

    doc.write('------SETTINGS------\n')
    for atr in inspect.getmembers(settings):
        if atr[0][0] != '_' and not inspect.ismethod(atr[1]):
            line = atr[0] + ' '*(14-len(atr[0])) + ' : ' + str(atr[1]) + '\n'
            doc.write(line)

    doc.write('\n-------VIDEO--------\n')
    for atr in inspect.getmembers(video):
        if atr[0][0] != '_' and not inspect.ismethod(atr[1]):
            if not atr[0] == 'Frames':
                line = atr[0] + ' '*(14-len(atr[0])) + ' : ' + str(atr[1]) + '\n'
                doc.write(line)
    doc.close()
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
    print()
    print('Sauvegarde de la vidéo en cours ...', end='')
    for frame in video.Frames:
        img = draw_cross_color(frame.array, frame, crosswidth)
        # img = Add_pas(img, pas)
        out.write(img)
    print('\rSauvegarde de la vidéo -------------------------------------------- OK', end='\n')
    return None

print()
main()