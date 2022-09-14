# Import modules
import pymediainfo as mi
import numpy as np
import cv2 as cv2
import csv as csv
import getpass as gp  # intégré à python par défaut
import os as os  # intégré à python par défaut
import sys as sys  # intégré à python par défaut
import shutil as sht  # intégré à python par défaut
import time as t  # intégré à python par défaut

class SettingError (Exception) :
    pass

def main():
    global definition, tol, minsize, maxdist, bordure_size, crosswidth, rectanglewidth

    # Réglages de rapidité/précision/sensibilité par défault.
    sys.setrecursionlimit(1000)
    definition = 1
    tol = 0.4

    print('\nInitialisation de la procédure')

    create_dir('bac')

    # On récupère notre vidéo
    videoinput()

    # On récupère des infos supplémentaires
    get_frames()
    get_framerate()
    modeinput()
    get_framessize()

    # delete_dir('bac')
    cinput()
    refinput()

    # On définit la taille des indicateurs visuels par rapport à la taille de l'image
    minsize = int(Framesize[1] / 300)
    maxdist = int(Framesize[1] / 10)
    bordure_size = int(Framesize[1] / 30)
    crosswidth = int(Framesize[1] / 500)
    rectanglewidth = int(Framesize[1] / 1250)

    # On traite la première frame seulement pour vérifier aue tous les reglages sont bons
    isOK = False
    while not isOK:
        calibration()
        if yn('Le traitement est-il bon ?'):
            isOK = True
        else:
#            if not yn('les repères sont bien de couleur ' + ['bleue', 'verte', 'rouge'][c] + ' ?'):
#                cinput()
#            else:
#                tol += float(input('\nTolérance actuelle : ' + str(tol) + ', implémenter de : '))
            verif_settings()

    # Une fois que tout est bon on traite la vidéo
    videotreatement()

    # On télécharge les données
    if yn("Voulez vous télécharger les résultats de l'étude ?"):
        reboot()
        add_subdata_dirs()
        videodownload()
        datadownload()
        framesdownload()
        create_video()

    print('\nProcédure terminée')

    return None


# paths gestion

user = gp.getuser()

paths = {}

paths['bac'] = '/Users/' + user + '/Desktop/bac'
paths['calib'] = '/Users/' + user + '/Documents/##calibdir##'
paths['video storage'] = '/Users/' + user + '/Documents/temporary storage.mp4'
paths['data'] = '/Users/' + user + '/Desktop/data'


def add_subdata_dirs():
    global video
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


# IHM

def videoinput():
    global video
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
        return None
    elif len(bac) == 1 and bac[0].split('.')[1] != 'mp4':
        print('Veuillez fournir une vidéo au format mp4')
        delete_dir('bac')
        videoinput()
    elif len(bac) > 1:
        print("Veuillez ne placer qu'un document dans le bac")
        delete_dir('bac')
        videoinput()


def cinput():
    global c
    while True:
        c = input('\nCouleur des repères à étudier (0=bleu, 1=vert, 2=rouge) : ')
        if c in ['0', '1', '2']:
            c = int(c)
            return None
        else:
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def refinput ():
    global lenscale
    while True:
        l = input('\nlongueur entre les deux premiers repères(cm) : ')
        try :
            lenscale = int(l)
            return None
        except ValueError :
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')


def modeinput():
    global mode
    while True:
        mode = input('\nLa vidéo est en mode (1=portrait, 0=landscape) : ')
        if mode in ['1', '0']:
            mode = int(mode)
            return None
        else:
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def verif_settings ():
    global tol
    print('1 orientation de la vidéo :', ['landscape', 'portrait'][mode])
    print('2 couleur des repères :', ['bleue', 'verte', 'rouge'][c])
    print('3 tolérance : ', tol)
    which = input('quel réglage vous semble-t-il éroné (0=aucun, 1, 2, 3) ? ')
    try :
        assert which in ['0', '1', '2', '3']
    except AssertionError :
        print ('vous devez avoir fait une erreur, veuillez réessayer')
    if which == '0' :
        pass
    elif which == '1' :
        modeinput()
    elif which == '2' :
        cinput()
    else :
        tol += float(input('\nTolérance actuelle : ' + str(tol) + ', implémenter de : '))


def yn(question):
    assert type(question) == str
    while True:
        yn = input('\n' + question + '\n[y]/n : ')
        if yn in ['y', '', 'n']:
            if yn == 'y' or yn == '':
                return True
            else:
                return False
        else:
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')


# Information recuperation tools

def get_frames():
    """
    Récupère l'ensembe des frames.
    Renvoie un dictionaire où les clés sont les numéros de frames et les valeurs des tableaux de type uint8.
    """
    global video, frames
    frames = {}
    cam = cv2.VideoCapture(paths['vidéoinput'])
    frame_number = 0
    print('\nRécupération de la vidéo en cours ...')
    while True:
        ret, frame = cam.read()
        if ret:
            frames['frame.' + str(frame_number)] = frame
            frame_number += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    print('\rRécupération de la vidéo ------------------------------------------ OK')
    return None


def get_framerate():
    """
    Renvoie dans le spectre global un dictionaire avec en clefs les numéros des frames et en valeurs des tableaux de
    type uint8.
    """
    global video, Framerate
    media_info = mi.MediaInfo.parse(paths['vidéoinput'])
    tracks = media_info.tracks
    for i in tracks:
        if i.track_type == 'Video':
            Framerate = round(float(i.frame_rate))
    return None


def get_framessize():
    """
    Renvoie dans le spectre global un tuple de deux valeurs : la hauteur et largeur des frames de la video.
    """
    global video, Framesize
    media_info = mi.MediaInfo.parse(paths['vidéoinput'])
    video_tracks = media_info.video_tracks[0]
    dim = [int(video_tracks.sampled_width), int(video_tracks.sampled_height)]
    if mode == 0:
        height = min(dim)
        width = max(dim)
    elif mode == 1:
        height = max(dim)
        width = min(dim)
    Framesize = (width, height)
    return None

def detScale (positions:dict, lenscale):
    global scale
    a = list(positions.keys())[0]
    b = list(positions.keys())[1]
    apos, bpos = positions[a], positions[b]
    xa , ya , xb, yb = apos[0], apos[1], bpos[0], bpos[1]
    scale = lenscale / ( ( (xa-xb)**2 + (ya-yb)**2 )**(1/2) )
    return None


# Frame preparation tools

def rate_rgb(pixel: list) -> float:
    """
    Calcul le poids relatif de la composante c du pixel pixel parmis les composantes rgb qui le définissent.
    pixel : élement de l'image d'origine sous la forme [r, g, b].
    c = 0(rouge), 1(vert) ou 2(bleu).
    """
    global c
    assert c in [0, 1, 2]
    # la rédaction ci-dessous n'est pas idéale, mais l'utilisation du np.sum rend le traitement trop long
    return int(pixel[c]) / (int(pixel[0]) + int(pixel[1]) + int(pixel[2]) + 1)


def prep(image):
    """
    Renvoie une image en noir et blanc
    image : image de depart.
    Definition : l'image finale contiendra 1/definition² pixels de l'image initiale. Attention les dimensions de l'image
    sont donc modifiées.
    """
    global definition
    assert 0 < definition
    assert type(definition) == int
    simplified_im = []
    h = len(image)
    w = len(image[0])
    for i in range(int(h / definition)):
        line = []
        for j in range(int(w / definition)):
            pixel = image[i * definition][j * definition]
            if rate_rgb(pixel) < tol:
                line.append(0)
            else:
                line.append(255)
        simplified_im.append(line)
    return simplified_im


# Treatement tools

def frametreatement(frame):
    """
    Permet le traitement de la frame passée en argument.
    frame : tableau uint8.
    """
    global definition
    isOK = False
    while not isOK and definition <= 15:
        try:
            NB_im = prep(frame)
            extremas = objects_identification(NB_im)
            isOK = True
        except RecursionError:
            print('\rDéfinition trop élevée, tentative avec une défintion plus faible', end='')
            definition += 1
            frametreatement(frame)

    if isOK:
        extremas = rectifyer(extremas)
        return extremas, NB_im
    else:
        raise SettingError


def videotreatement():
    """
    Permet le traitement de l'ensemble des frames qui constituent la vidéo ainsi que le suivi des objets
    """
    global video, frames, positions, maxdist, bordure_size, tracked_objects
    # positions = {}
    tracked_objects = {}
    obj_compteur = 0
    frames_keys = list(frames.keys())

    print('')

    # Initialisation
    tracked_objects[frames_keys[0]] = {}
    for obj in positions[frames_keys[0]]:
        tracked_objects[frames_keys[0]]['obj-' + str(obj_compteur)] = positions[frames_keys[0]][obj]
        obj_compteur += 1

    bande1 = [i for i in range(0, bordure_size + 1)]
    bande2 = [i for i in range(Framesize[1] - bordure_size, Framesize[1] + 1)]

    for i in range(1, len(frames_keys)):
        tracked_objects[frames_keys[i]] = {}

        treated = frametreatement(frames[frames_keys[i]])[0]
        positions[frames_keys[i]] = position(treated)

        for obj1 in positions[frames_keys[i]]:

            identified = False
            distances_list = {}
            x1, y1 = positions[frames_keys[i]][obj1][0], positions[frames_keys[i]][obj1][1]

            for obj2 in tracked_objects[frames_keys[i - 1]]:
                x2, y2 = tracked_objects[frames_keys[i - 1]][obj2][0], tracked_objects[frames_keys[i - 1]][obj2][1]
                d = round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2), 2)
                distances_list[obj2] = d

            if len(distances_list) != 0:
                min_key = min(distances_list, key=distances_list.get)
                distance = distances_list[min_key]
                if distance < maxdist:
                    identified = True
                    tracked_objects[frames_keys[i]][min_key] = positions[frames_keys[i]][obj1]

            if not identified:
                if x1 in bande1 or x1 in bande2:
                    tracked_objects[frames_keys[i]]['obj-' + str(obj_compteur)] = [x1, y1]
                    obj_compteur += 1
                if y1 in bande1 or y1 in bande2:
                    tracked_objects[frames_keys[i]]['obj-' + str(obj_compteur)] = [x1, y1]
                    obj_compteur += 1

        progression = round((int(frames_keys[i].split('.')[1]) / (len(frames) - 1)) * 100, 1)
        print('\rTraitement de ' + video + ' en cours :', str(progression), '%', end='')
        t.sleep(.02)

    print('\nTraitement de ' + video + ' -------------------------------------------- Finit')
    return None


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


def objects_identification(image) -> dict:
    """
    Regroupe tout les objets de l'image dans un dictionnaire.
    image : image en N&B sous la forme d'un array de 0 et 255.
    """
    h = len(image)
    w = len(image[0])
    objects = {}
    extremas = {}
    n = 0
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


def rectifyer(extremas: dict) -> dict:
    """
    Rectifie quelques erreurs.
    """
    # On supprime les objets trop petits, probablement issus d'erreurs.
    global minsize
    problematic_objects = []
    for obj in extremas:
        if extremas[obj][2] - extremas[obj][0] < minsize or extremas[obj][3] - extremas[obj][1] < minsize:
            problematic_objects.append(obj)
    for obj in problematic_objects:
        del extremas[obj]
    # On renome nos objets.
    i = 0
    dico2 = {}
    for obj in extremas:
        dico2['obj-' + str(i)] = extremas[obj]
        i += 1
    return dico2


# Rectangles/cross drawing tools
def copy_im (image):
    L = len(image)
    l = len(image[0])
    newIm = []
    for y in range (L):
        newLine = []
        for x in range(l):
            newLine.append(image[y][x])
        newIm.append(newLine)
    return newIm

def Add_rectangle_NB(image, extremas):
    global rectanglewidth
    L = len(image)
    l = len(image[0])
#    NewIm = copy_im(image)
    for key in extremas:
        xmin, ymin, xmax, ymax = int(extremas[key][0]), int(extremas[key][1]), int(extremas[key][2]), int(
            extremas[key][3])
        for i in range(xmin - rectanglewidth, xmax + rectanglewidth + 1):
            for n in range(rectanglewidth + 1):
                image[(ymin - n) % L][i % l], image[(ymax + n) % L][i % l] = 255, 255
        for j in range(ymin - rectanglewidth, ymax + rectanglewidth + 1):
            for n in range(rectanglewidth + 1):
                image[j % L][(xmin - n) % l], image[j % L][(xmax + n) % l] = 255, 255
    return image

def Add_cross_color(image, positions):
    global crosswidth
    L = len(image)
    l = len(image[0])
#    NewIm = copy_im(image)
    for obj in positions:
        x = int(positions[obj][0])
        y = int(positions[obj][1])
        for i in range(x - crosswidth * 10, x + crosswidth * 10 + 1):
            for n in range(y - int(crosswidth / 2), y + int(crosswidth / 2) + 1):
                image[n % L][i % l] = [0, 255, 0]
        for j in range(y - crosswidth * 10, y + crosswidth * 10 + 1):
            for n in range(x - int(crosswidth / 2), x + int(crosswidth / 2) + 1):
                image[j % L][n % l] = [0, 255, 0]
    return image

def Add_scale(image):
    global scale, crosswidth, bordure_size
    L = len(image)
    l = len(image[0])
    for i in range (int(10/scale)):
        for j in range (crosswidth):
            image[(j+L-bordure_size) % L][(bordure_size+i) % l] = [0, 0, 255]
    cv2.putText(image, '10cm', (bordure_size, L-bordure_size-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    return image

# Calibration fcts

def calibration():
    """
    À effectuer avant le traitement de l'ensemble de la vidéo pour vérifier le bon réglage de l'ensmeble des paramètres.
    """
    global video, frames, Framesize, lenscale, positions, scale
    positions = {}

    print('\nTraitement en cours ...')
    first_key = list(frames.keys())[0]
    first = frames[first_key]

    try :
        detected = frametreatement(first)
    except SettingError :
        print('\nLa tolérance doit être mal réglée, vérifiez le réglage')
        return None

    extremas = detected[0]
    positions[first_key] = position(rectifyer(detected[0]))
    detScale(positions[first_key], lenscale)

    print('\nTraitement -------------------------------------------------------- OK')

    images_names = []
    create_dir('calib')

    color_im = first
    images_names.append('color_im')
    fill_calibdir(color_im, 'color_im')

    NB_im = cv2.resize(np.uint8(detected[1]), Framesize)
    images_names.append('NB_im')
    fill_calibdir(NB_im, 'NB_im')

    treated_NB = np.uint8(Add_rectangle_NB(NB_im, extremas))
    images_names.append('treated_NB')
    fill_calibdir(treated_NB, 'treated_NB')

    ImWithCross = Add_cross_color(color_im, positions[first_key])
    ImWithScale = Add_scale(ImWithCross)
    treated_color = np.uint8(ImWithScale)
    images_names.append('treated_color')
    fill_calibdir(treated_color, 'treated_color')

    print("\nAffichage du résultat, veuillez checker sa correction\n(une fenêtre a dû s'ouvrir)")
    calib_show(images_names)
    print('Validation du résultat -------------------------------------------- OK')

    sht.rmtree(paths['calib'])

    return None


def fill_calibdir(image, image_name):
    cv2.imwrite(paths['calib'] + '/' + image_name + '.jpg', image)
    return None


def calib_show(images_names: list):
    for i in range(len(images_names)):
        cv2.imshow('Config Window - ' + images_names[i], cv2.imread(paths['calib'] + '/' + images_names[i] + '.jpg'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return None


# Data download fcts

def videodownload():
    global video
    create_dir('vidéodl')
    source = paths['video storage']
    destination = paths['vidéodl'] + '/vidéo' + '.mp4'
    sht.copy2(source, destination)
    os.remove(paths['video storage'])
    return None


def datadownload():
    global video, tracked_objects, Framerate
    create_dir('csv')
    print('\nSauvegarde de la data en cours ...')
    nom_colonnes = ['frame', 'time']
    objects = []
    for frame in tracked_objects:
        for obj in tracked_objects[frame]:
            if obj not in objects:
                objects.append(obj)
                nom_colonnes += ['X' + str(obj), 'Y' + str(obj)]
    dos = open(paths['csv'] + '/positions objets.csv', 'w')
    array = csv.DictWriter(dos, fieldnames=nom_colonnes)
    array.writeheader()
    for frame in tracked_objects:
        dico = {'frame': frame, 'time': round(int(frame.split('.')[1]) / Framerate, 3)}
        for obj in tracked_objects[frame]:
            dico['X' + str(obj)] = tracked_objects[frame][obj][0]
            dico['Y' + str(obj)] = tracked_objects[frame][obj][1]
        array.writerow(dico)
    dos.close()
    print('Sauvegarde de la data --------------------------------------------- OK')
    return None


def framesdownload():
    global video, frames, tracked_objects
    create_dir('non treated frames')
    create_dir('treated frames')
    print('\nSauvegarde des frames en cours ...')
    for frame in frames:
        name = paths['non treated frames'] + '/frame' + str(int(frame.split('.')[1])) + '.jpg'
        cv2.imwrite(name, frames[frame])
        name = paths['treated frames'] + '/frame' + str(int(frame.split('.')[1])) + '.jpg'
        cv2.imwrite(name, np.uint8(cross_color(frames[frame], tracked_objects[frame])))
    print('Sauvegarde des frames --------------------------------------------- OK')
    return None


def create_video():
    global frames, Framerate, Framesize, tracked_objects
    out = cv2.VideoWriter(paths['vidéodl'] + '/vidéo traitée' + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), Framerate,
                          Framesize)
    print('\nSauvegarde de la vidéo en cours ...')
    for frame in frames:
        img = np.uint8(cross_color(frames[frame], tracked_objects[frame]))
        out.write(img)
    print('Sauvegarde de la vidéo -------------------------------------------- OK')
    return None


# Reboot

def reboot():
    try:
        delete_dir('csv')
        delete_dir('frames')
        delete_dir('vidéodl')
    except KeyError:
        pass
    return None


# Execution

main()
