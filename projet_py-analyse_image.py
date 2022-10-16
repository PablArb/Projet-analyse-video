import numpy as np
import cv2 as cv2
import csv as csv
import pymediainfo as mi
import os as os         # intégré à python par défaut
import sys as sys       # intégré à python par défaut
import time as t        # intégré à python par défaut
import getpass as gp    # intégré à python par défaut
import shutil as sht    # intégré à python par défaut

class SettingError (Exception):
    pass

class Break (Exception):
    pass


# main
def main():
    """
    """
    # Réglages de rapidité/précision/sensibilité par défault.
    sys.setrecursionlimit(1000)
    definition = 1
    tol = 0.4

    print('\nInitialisation de la procédure')

    create_dir('bac')
    try:
        video = Video(videoinput())
    except Break:
        print('\nFin de la procédure')
        return None
    delete_dir('bac')

    try:
        c = cinput()
        mode = get_mode(video, video.Framessize)
        lenref = refinput()
    except Break:
        print('\nFin de la procédure')
        return None

    # On définit la taille des indicateurs visuels par rapport à la taille de l'image
    minsize = int(video.Framessize[1] / 300)
    maxdist = int(video.Framessize[1] / 10)
    bordure_size = int(video.Framessize[1] / 30)
    crosswidth = int(video.Framessize[1] / 500)
    rectanglewidth = int(video.Framessize[1] / 1250)


    # On traite la première frame seulement pour vérifier aue tous les reglages sont bons
    try :
        isOK = False
        while not isOK:
            calibration(video, definition, tol, c, minsize, crosswidth, rectanglewidth, bordure_size, lenref)
            if yn('Le traitement est-il bon ?'):
                isOK = True
            else:
                tol, c = verif_settings(video, tol, c, video.mode)
                definition = 1
    except Break :
        print ('\nFin de la procédure')
        return None

    # Une fois que tout est bon on traite la vidéo
    videotreatement(video, tol, c, minsize, crosswidth, rectanglewidth, bordure_size, maxdist)

    # On télécharge les données
    if yn("Voulez vous télécharger les résultats de l'étude ?"):
        resultsdownload(video, video.scale, crosswidth)
    print('\nProcédure terminée')

    return None



# path gestion

user = gp.getuser()
paths = {}
L_paths = ['bac', 'calib', 'video storage', 'data']
paths[L_paths[0]] = '/Users/' + user + '/Desktop/bac'
paths[L_paths[1]] = '/Users/' + user + '/Desktop/.##calibdir##'
paths[L_paths[2]] = '/Users/' + user + '/Desktop/.##temporary storage##.mp4'
paths[L_paths[3]] = '/Users/' + user + '/Desktop/data'

if os.name == 'nt':
    for el in L_paths:
        paths[el] = 'C:'+paths[el]

def add_subdata_dirs(video):
    paths['csv'] = paths['data'] + '/' + video + '/csv'
    paths['vidéodl'] = paths['data'] + '/' + video + '/vidéo'
    paths['frames'] = paths['data'] + '/' + video + '/frames'
    paths['treated frames'] = paths['frames'] + '/treated'
    paths['non treated frames'] = paths['frames'] + '/non treated'
    return None

def create_dir(dir: str):
    p = paths[dir]
    try:
        if not os.path.exists(p):
            os.makedirs(p)
    except OSError:
        print('Error: Creating directory of data')
    return None

def delete_dir(dir: str):
    p = paths[dir]
    try:
        if os.path.exists(p):
            sht.rmtree(p)
    except OSError:
        print('Error: Creating directory of data')
    return None



# video constructor

class Video:
    def __init__(self, id):
        self.id = id
        self.frames = get_frames(self)
        self.Framerate = get_framerate(self)
        self.Framessize = get_framessize()
        self.mode = None
        self.scale = None


class Frame:
    def __init__(self, id, array):
        self.id = id
        self.array = array
        self.identified_objects = {}

def get_framerate(video):
    """
    Renvoie dans le spectre global un dictionaire avec en clefs les numéros des frames et en valeurs des tableaux de
    type uint8.
    """
    media_info = mi.MediaInfo.parse(paths['vidéoinput'])
    tracks = media_info.tracks
    for i in tracks:
        if i.track_type == 'Video':
            Framerate = float(i.frame_rate)
    video.Framerate = Framerate
    return Framerate

def get_framessize():
    """
    Renvoie dans le spectre global un tuple de deux valeurs : la hauteur et largeur des frames de la video.
    """
    media_info = mi.MediaInfo.parse(paths['vidéoinput'])
    video_tracks = media_info.video_tracks[0]
    Framessize = [int(video_tracks.sampled_width), int(video_tracks.sampled_height)]
    return Framessize

def get_mode(video, Framessize):
    # Framessize = get_framessize()
    while True:
        mode = input('\nLa vidéo est en mode (1=landscape, 2=portrait) : ')
        if mode in ['1', '2', 'break']:
            if mode == '1':
                height = min(Framessize)
                width = max(Framessize)
            elif mode == '2':
                height = max(Framessize)
                width = min(Framessize)
            else:
                raise Break
            Framessize = (width, height)
            video.Framessize = Framessize
            video.mode = int(mode)
            return Framessize, int(mode)
        else:
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def get_frames(video):
    """
    Récupère l'ensembe des frames.
    Renvoie un dictionaire où les clés sont les numéros de frames et les valeurs des tableaux de type uint8.
    """
    frames = []
    cam = cv2.VideoCapture(paths['vidéoinput'])
    frame_number = 0
    print('\nRécupération de la vidéo en cours ...')
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
    video.frames = frames
    return frames

def detScale (video, positions:dict, lenref):
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



# Treatement tools

def videotreatement(video, tol, c, minsize, crosswidth, rectanglewidth, bordure_size, maxdist):
    """
    Permet le traitement de l'ensemble des frames qui constituent la vidéo ainsi que le suivi des objets
    """
    global positions, definition
    frames = video.frames
    obj_compteur = 0
    T = t.time()

    print('')

    # Initialisation
    for obj in positions[frames[0].id]:
        video.frames[0].identified_objects['obj-' + str(obj_compteur)] = positions[frames[0].id][obj]
        obj_compteur += 1

    bande1 = [i for i in range(0, bordure_size + 1)]
    bande2 = [i for i in range(video.Framessize[1] - bordure_size, video.Framessize[1] + 1)]

    for i in range(1, len(frames)):
        try :
            treated = frametreatement(frames[i].array, tol, c, minsize)[0]
            positions[frames[i].id] = position(treated)

            for obj1 in positions[frames[i].id]:

                identified = False
                distances_list = {}
                x1, y1 = positions[frames[i].id][obj1][0], positions[frames[i].id][obj1][1]

                for obj2 in video.frames[i-1].identified_objects:
                    x2, y2 = video.frames[i-1].identified_objects[obj2][0], \
                             video.frames[i-1].identified_objects[obj2][1]
                    d = round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2), 2)
                    distances_list[obj2] = d

                if len(distances_list) != 0:
                    min_key = min(distances_list, key=distances_list.get)
                    distance = distances_list[min_key]
                    if distance < maxdist:
                        identified = True
                        video.frames[i].identified_objects[min_key] = positions[frames[i].id][obj1]

                if not identified:
                    if x1 in bande1 or x1 in bande2:
                        video.frames[i].identified_objects['obj-' + str(obj_compteur)] = [x1, y1]
                        obj_compteur += 1
                    if y1 in bande1 or y1 in bande2:
                        video.frames[i].identified_objects['obj-' + str(obj_compteur)] = [x1, y1]
                        obj_compteur += 1
        except SettingError :
            pass

        if t.time() - T > 0.5 :
            progression = round((int(frames[i].id.split('.')[1]) / (len(frames) - 1)) * 100, 1)
            print('\rTraitement de ' + video.id + ' en cours :', str(progression), '%', end='')
            T = t.time()

    print('\nTraitement de ' + video.id + ' -------------------------------------------- Finit')
    return None

def frametreatement(frame, tol, c, minsize):
    """
    Permet le traitement de la frame passée en argument.
    frame : tableau uint8.
    """
    global definition
    isOK = False
    while not isOK and definition <= 15:
        try:
            NB_im = prep(frame, definition, tol, c)
            extremas = objects_identification(NB_im, definition)
            isOK = True
        except RecursionError:
            print('\rDéfinition trop élevée, tentative avec une défintion plus faible', end='')
            definition += 1
            frametreatement(frame, tol, c, minsize)

    if isOK:
        extremas = rectifyer(extremas, minsize)
        return extremas, NB_im
    else:
        raise SettingError

# Frame manipulation tools

def get_neighbours(image, pixel: list) -> list:
    """
    Renvoie la liste des voisins du pixel 'pixel' à étudier dans le cadre de la recherche d'objet.
    image : image en N&B.
    pixel : sous la forme [j,i].
    """
    x, y = pixel[0], pixel[1]
    h = len(image)
    w = len(image[0])
    L_neighours_to_test = [[(x - 1) % w, (y - 1) % h], [(x - 1) % w, y], [(x - 1) % w, (y + 1) % h],
                           [x, (y - 1) % h], [x, (y + 1) % h],
                           [(x + 1) % w, (y - 1) % h], [(x + 1) % w, y], [(x + 1) % w, (y + 1) % h]]
    L_neighours = []
    for element in L_neighours_to_test:
        if image[element[1]][element[0]] == 255:
            L_neighours.append(element)
    return L_neighours

def visiter(image, depart: list, object: list, extr: list) -> list:
    """
    Regroupe tous les pixels appartenant a un même objets (forme blanche ici) sous la forme d'une liste.
    image : image en N&B.
    depart : pixel duquel on va partir pour 'explorer' notre objet, sous la forme [j,i].
    objet : liste contenant tout les pixels appartenants au même objet.
    """
    if depart not in object:
        object.append(depart)
        # xmin, ymin, xmax, ymax = extr[0], extr[1], extr[2], extr[3] (pour info)
        if depart[0] < extr[0]:
            extr[0] = depart[0]
        if depart[1] < extr[1]:
            extr[1] = depart[1]
        if depart[0] > extr[2]:
            extr[2] = depart[0]
        if depart[1] > extr[3]:
            extr[3] = depart[1]
    for pixel in get_neighbours(image, depart):
        if pixel not in object:
            visiter(image, pixel, object, extr)
    return object, extr

def discovery(image, depart: list) -> list:
    object = [depart]
    init_extr = [depart[0], depart[1], depart[0], depart[1]]
    infos = visiter(image, depart, object, init_extr)
    object = infos[0]
    extr = infos[1]
    return object, extr

def objects_identification(image, definition) -> dict:
    """
    Regroupe tout les objets de l'image dans un dictionnaire.
    image : image en N&B sous la forme d'un array de 0 et 255.
    """
    h = len(image)
    w = len(image[0])
    objects = {}
    extremas = {}
    n = 0
    # pas = int(1 + 10/definition)
    for j in range(h):
        for i in range(w):
            if image[j][i] == 255:
                element_in = False
                for obj in objects:
                    if [i, j] in objects[obj]:
                        element_in = True
                if not element_in:
                    infos = discovery(image, [i, j])
                    objects[n] = infos[0]
                    extremas[n] = infos[1]
                    n += 1
    for obj in extremas:
        xmin, ymin, xmax, ymax = extremas[obj][0], extremas[obj][1], extremas[obj][2], extremas[obj][3]
        extremas[obj] = [xmin * definition, ymin * definition, xmax * definition, ymax * definition]
    return extremas

def position(extremas: dict) -> list:
    """
    Récupère la position d'un objet à partir des extremas.
    Renvoie un dictionnaire où les clefs sont les noms des ifférents objets détectés sur la frame étudiée et les valeurs
    sont les coordonées du 'centre' de l'objet.
    """
    position = {}
    for obj in extremas:
        x = (extremas[obj][0] + extremas[obj][2]) / 2
        y = (extremas[obj][1] + extremas[obj][3]) / 2
        position[obj] = [x, y]
    return position

def rectifyer(extremas: dict, minsize) -> dict:
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


def rate_rgb(pixel: list, c) -> float:
    """
    Calcul le poids relatif de la composante c du pixel pixel parmis les composantes rgb qui le définissent.
    pixel : élement de l'image d'origine sous la forme [r, g, b].
    c = 0(rouge), 1(vert) ou 2(bleu).
    """
    assert c in [0, 1, 2]
    # la rédaction ci-dessous n'est pas idéale, mais l'utilisation du np.sum rend le traitement trop long
    return int(pixel[c]) / (int(pixel[0]) + int(pixel[1]) + int(pixel[2]) + 1)

def prep(image, definition, tol, c):
    """
    Renvoie une image en noir et blanc
    image : image de depart.
    Definition : l'image finale contiendra 1/definition² pixels de l'image initiale. Attention les dimensions de l'image
    sont donc modifiées.
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
    return simplified_im


# Calibration fcts

def calibration(video, definition2, tol, c, minsize, crosswidth, rectanglewidth, bordure_size, lenref):
    """
    À effectuer avant le traitement de l'ensemble de la vidéo pour vérifier le bon réglage de l'ensmeble des paramètres.
    """
    global positions, definition
    definition = definition2
    positions = {}

    print('\nTraitement en cours ...')
    first = copy_im(video.frames[0].array)

    try :
        detected = frametreatement(first, tol, c, minsize)
    except SettingError :
        print('\nIl y a un problème, veuillez vérifiez les réglages')
        verif_settings()
        definition = 1
        calibration()
        return None

    extremas = detected[0]
    positions[video.frames[0].id] = position(rectifyer(detected[0], minsize))
    scale = detScale(video, positions[video.frames[0].id], lenref)

    print('\nTraitement -------------------------------------------------------- OK')

    images_names = []
    create_dir('calib')

    color_im = first
    images_names.append('color_im')
    fill_calibdir(color_im, 'color_im')

    NB_im = cv2.resize(np.uint8(detected[1]), video.Framessize)
    images_names.append('NB_im')
    fill_calibdir(NB_im, 'NB_im')

    treated_NB = np.uint8(rectangle_NB(NB_im, extremas, rectanglewidth))
    images_names.append('treated_NB')
    fill_calibdir(treated_NB, 'treated_NB')

    ImWithCross = cross_color(color_im, positions[video.frames[0].id], crosswidth)
    ImWithScale = Add_scale(ImWithCross, scale,crosswidth, bordure_size)
    treated_color = np.uint8(ImWithScale)
    images_names.append('treated_color')
    fill_calibdir(treated_color, 'treated_color')

    print("\nAffichage du résultat, veuillez checker sa correction\n(une fenêtre a dû s'ouvrir)")
    calib_show(images_names)
    print('Validation du résultat -------------------------------------------- OK')

    sht.rmtree(paths['calib'])
    return None

def copy_im (image):
    L = len(image)
    l = len(image[0])
    newIm = []
    for y in range (L):
        newLine = []
        for x in range(l):
            newLine.append(image[y][x])
        newIm.append(newLine)
    return np.uint8(newIm)

def fill_calibdir(image, image_name):
    cv2.imwrite(paths['calib'] + '/' + image_name + '.jpg', image)
    return None

def calib_show(images_names: list):
    for i in range(len(images_names)):
        cv2.imshow('Config Window - ' + images_names[i], cv2.imread(paths['calib'] + '/' + images_names[i] + '.jpg'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return None



# indicators

def rectangle_NB(image, extremas, rectanglewidth):
    L = len(image)
    l = len(image[0])
    marge = 4
    for key in extremas:
        xmin, ymin, xmax, ymax = int(extremas[key][0])-marge, int(extremas[key][1])-marge, int(extremas[key][2])+marge,\
                                 int(extremas[key][3])+marge
        for i in range(xmin - rectanglewidth, xmax + rectanglewidth + 1):
            for n in range(rectanglewidth + 1):
                image[(ymin - n) % L][i % l], image[(ymax + n) % L][i % l] = 255, 255
        for j in range(ymin - rectanglewidth, ymax + rectanglewidth + 1):
            for n in range(rectanglewidth + 1):
                image[j % L][(xmin - n) % l], image[j % L][(xmax + n) % l] = 255, 255
    return np.uint8(image)

def cross_color(image, positions, crosswidth):
    L = len(image)
    l = len(image[0])
    for obj in positions:
        x = int(positions[obj][0])
        y = int(positions[obj][1])
        for i in range(x - crosswidth * 10, x + crosswidth * 10 + 1):
            for n in range(y - int(crosswidth / 2), y + int(crosswidth / 2) + 1):
                image[n % L][i % l] = [0, 255, 0]
        for j in range(y - crosswidth * 10, y + crosswidth * 10 + 1):
            for n in range(x - int(crosswidth / 2), x + int(crosswidth / 2) + 1):
                image[j % L][n % l] = [0, 255, 0]
    return np.uint8(image)

def Add_scale(image, scale, crosswidth, bordure_size):
    L = len(image)
    l = len(image[0])
    for i in range(int(1/scale)):
        for j in range(crosswidth):
            image[(j+L-bordure_size) % L][(bordure_size+i) % l] = [255, 255, 255]
    cv2.putText(image, '1cm', (bordure_size, L-bordure_size-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    return image


def videoinput():
    isempty = True
    print('\nPlacez la vidéo (.mp4) à étudier dans le bac sur votre bureau.')
    while isempty:
        if len(os.listdir(paths['bac'])) != 0:
            isempty = False
        t.sleep(0.5)
    bac = os.listdir(paths['bac'])
    if len(bac) == 1 and bac[0].split('.')[1] == 'mp4':
        video = bac[0].split('.')[0]
        paths['vidéoinput'] = paths['bac'] + '/' + video + '.mp4'
        sht.copy2(paths['vidéoinput'], paths['video storage'])
        return video
    elif len(bac) == 1 and bac[0].split('.')[1] != 'mp4':
        print('Veuillez fournir une vidéo au format mp4')
        delete_dir('bac')
        videoinput()
    elif len(bac) > 1:
        print("Veuillez ne placer qu'un document dans le bac")
        delete_dir('bac')
        videoinput()

def cinput():
    while True :
        c = input('\nCouleur des repères à étudier (1=bleu, 2=vert, 3=rouge) : ')
        if c in ['1', '2', '3', 'break']:
            if c in ['1', '2', '3']:
                c = int(c)-1
                return c
            else:
                raise Break
        else:
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def refinput ():
    while True:
        l = input('\nlongueur entre les deux premiers repères(cm) : ')
        try :
            if l == 'break':
                raise Break
            lenref = float(l)
            return lenref
        except ValueError :
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def verif_settings (video, tol, c, mode):
    while True :
        print('\n1 orientation de la vidéo :', ['landscape', 'portrait'][mode-1])
        print('2 couleur des repères :', ['bleue', 'verte', 'rouge'][c])
        print('3 tolérance : ', tol)
        which = input('quel réglage vous semble-t-il éroné (0=aucun, 1, 2, 3) ? ')
        if which in ['0', '1', '2', '3', 'pres', 'break']:
            if which == '0':
                return tol, c
            elif which == '1':
                get_mode(video)[1]
                return tol, c
            elif which == '2':
                c = cinput()
                return tol, c
            elif which == '3':
                tol += float(input('\nTolérance actuelle : ', tol, ', implémenter de : '))
                return tol, c
            elif which == 'pres':
                sys.setrecursionlimit(int(input('setrecursionlimit : ')))
                return tol, c
            else :
                raise Break
        else:
            print ('vous devez avoir fait une erreur, veuillez réessayer')

def yn(question):
    assert type(question) == str
    while True:
        yn = input('\n' + question + '\n[y]/n : ')
        if yn in ['y', '', 'n', 'break']:
            if yn == 'y' or yn == '':
                return True
            elif yn == 'n':
                return False
            else:
                raise Break
        else:
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')


def resultsdownload(video, scale, crosswidth):
    reboot(video)
    videodownload()
    datadownload(video, scale)
    # framesdownload(video, crosswidth)
    create_video(video, crosswidth)
    return None

def reboot(video):
    add_subdata_dirs(video.id)
    delete_dir('csv')
    delete_dir('frames')
    delete_dir('vidéodl')
    add_subdata_dirs(video.id)
    return None

def videodownload():
    create_dir('vidéodl')
    source = paths['video storage']
    destination = paths['vidéodl'] + '/vidéo' + '.mp4'
    sht.copy2(source, destination)
    os.remove(paths['video storage'])
    return None

def datadownload(video, scale):
    create_dir('csv')
    print('\nSauvegarde de la data en cours ...')
    nom_colonnes = ['frame', 'time']
    objects = []
    frames = video.frames
    for i in range (len(frames)):
        for obj in video.frames[i].identified_objects:
            if obj not in objects:
                objects.append(obj)
                nom_colonnes += ['X' + str(obj), 'Y' + str(obj)]
    dos = open(paths['csv'] + '/positions objets.csv', 'w')
    array = csv.DictWriter(dos, fieldnames=nom_colonnes)
    array.writeheader()
    for i in range (len(frames)):
        dico = {'frame': frames[i].id, 'time': round(int(frames[i].id.split('.')[1]) / video.Framerate, 3)}
        for obj in video.frames[i].identified_objects:
            dico['X' + str(obj)] = scale * video.frames[i].identified_objects[obj][0]
            dico['Y' + str(obj)] = scale * video.frames[i].identified_objects[obj][1]
        array.writerow(dico)
    dos.close()
    print('Sauvegarde de la data --------------------------------------------- OK')
    return None

def framesdownload(video, crosswidth):
    create_dir('non treated frames')
    create_dir('treated frames')
    print('\nSauvegarde des frames en cours ...')
    for frame in video.frames:
        name = paths['non treated frames'] + '/frame' + str(int(frame.id.split('.')[1])) + '.jpg'
        cv2.imwrite(name, frame.array)
        name = paths['treated frames'] + '/frame' + str(int(frame.id.split('.')[1])) + '.jpg'
        cv2.imwrite(name, cross_color(frame.array, frame.identified_objects, crosswidth))
    print('Sauvegarde des frames --------------------------------------------- OK')
    return None

def create_video(video, crosswidth):
    out = cv2.VideoWriter(paths['vidéodl'] + '/vidéo traitée' + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video.Framerate, video.Framessize)
    print('\nSauvegarde de la vidéo en cours ...')
    for frame in video.frames:
        img = np.uint8(cross_color(frame.array, frame.identified_objects, crosswidth))
        # img = frame.array
        out.write(img)
    print('Sauvegarde de la vidéo -------------------------------------------- OK')
    return None

main()