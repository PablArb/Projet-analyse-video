import numpy as np
import cv2
import sys                  # intégré à python par défaut
import time as t            # intégré à python par défaut
from IHM import visu, download, interact
from ERRORS import Break, SettingError
from VideoTreatment import Video, Object, Settings


# Calibration fcts

def calibration():
    """
    À effectuer avant le traitement de l'ensemble de la vidéo pour vérifier le
    bon réglage de l'ensmeble des paramètres.
    """
    global video, settings

    print()
    print('Traitement en cours ...', end='')
    first = visu.copy_im(video.Frames[0].array)

    try :

        detected = frametreatement(first, 0)
        extremas = rectifyer(detected[0], settings.minsize)
        positions = position(extremas)

        settings.step = Pas(extremas, settings.definition)
        detScale(video.lenref, positions)

    except SettingError :
        print('\rIl y a un problème, veuillez vérifiez les réglages', end='\n' )
        interact.verif_settings(video, settings)
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

    rectanglewidth = settings.rectanglewidth
    crosswidth = settings.crosswidth
    markerscolor =  video.markerscolor
    scale = video.scale
    frame = video.Frames[0]

    print('Création des visuels en cours ...', end='')
    visualisations = []

    color_im = visu.copy_im(first)
    visualisations.append(color_im)

    NB_im = reducer(color_im, settings.definition)
    NB_im = visu.reduced(video, settings, NB_im, rate_rgb)
    NB_im = cv2.resize(NB_im, video.Framessize)
    visualisations.append(NB_im)

    treated_NB = visu.copy_im(NB_im)
    treated_NB = visu.detection(treated_NB, detected[1])
    treated_NB = visu.rectangle_NB(treated_NB, extremas, rectanglewidth)
    visualisations.append(treated_NB)

    pos = [obj.positions[frame.id] for obj in frame.identifiedObjects]
    treated_color = visu.cross_color(first, pos, crosswidth, copy=True)
    treated_color = visu.scale(treated_color, scale, crosswidth, markerscolor)
    visualisations.append(treated_color)

    for im in visualisations :
        cv2.imshow('calibration window', im)
        print("\rAffichage du résultat (une fenêtre a dû s'ouvrir)", end='')
        cv2.waitKey(0)
        cv2.destroyWindow('calibration window')
        cv2.waitKey(1)

    t.sleep(0.1)
    print('\rValidation du résultat -------------------------------------------- OK', end='\n')
    t.sleep(0.1)

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

    for i in range(1, len(frames)): # frame 0 traitée durant l'initialisation
        try :

            markers_extremums = frametreatement(frames[i].array, i)[0]
            markers_extremums = rectifyer(markers_extremums, minsize)
            positions = position(markers_extremums)

            object_tracker(video, i, positions, maxdist, bordure_size)

        except SettingError :
            print('\rproblèmes dans les réglages')

        if t.time() - T >= 1 :
            progr = (int(frames[i].id.split('.')[1]) / (len(frames) - 1)) * 100
            progr = str(round(progr))
            tleft = waiting_time(i, len(frames), Ti)
            print('\033[2K\033[1GTraitement en cours : ' +progr+ ' % (' +tleft+ ')', end='')
            T = t.time()

    t.sleep(0.1)
    print('\rTraitement de ' + video.name + ' ' + '-'*( 67-15-len(video.name) ) + ' OK', end='\n\n')

    video.computationDuration = round(t.time()-Ti, 1)

    return None


def object_tracker(video, i, positions, maxdist, bordure_size):
    frames = video.Frames

    for obj1 in positions :

        identified = False
        distances_list = {}
        x1, y1 = positions[obj1][0], positions[obj1][1]

        for obj2 in frames[i-1].identifiedObjects:

            x2 = obj2.positions[frames[i-1].id][0]
            y2 = obj2.positions[frames[i-1].id][1]

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
    # Les objets apparaissant aux bordures de l'écran ne seront pas considérés
    # comme des erreurs mais comme des nouveaux objets entrant dans le chant de
    # la caméra.

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


def frametreatement(frame:np.array, i:int) -> tuple:
    """
    frame : image à traiter (tableau uint8).
    i : numméro de la frame que l'on traite.
    Permet le traitement de la frame passée en argument.
    """
    global settings
    isOK = False
    while not isOK and settings.definition <= settings.maxdef :
        try:
            image = reducer(frame, settings.definition)
            extremas, borders = objects_identification(image, i)
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

def objects_identification(image:np.array, i:int) -> tuple :
    """
    image : frame à traiter.
    i : indice de l'image à traiter.
    """

    global video, settings
    global at_border
    pas, tol = settings.step, settings.tol
# =============================================================================
#     maxdist = settings.maxdist
# =============================================================================
    markerscolor = video.markerscolor
    h = len(image)
    w = len(image[0])
    extremas = {}
    borders = {}
    n = 0

# =============================================================================
#     if i > 0:
#         PrevFrame = video.Frames[i-1]
#         AreasToExplore = []
#         for obj in PrevFrame.identifiedObjects :
#             [x, y] = obj.positions[PrevFrame.id]
#             if x - maxdist < 0 : xmin = 0
#             else : xmin = x-maxdist
#             if y - maxdist < 0 : ymin = 0
#             else : ymin = y - maxdist
#             NewArea = []
# =============================================================================


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
    return int(pixel[c]) / (int(pixel[0]) + int(pixel[1]) + int(pixel[2]) + 1)


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
        return 1
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


def cleaner(video:Video):
    sys.setrecursionlimit(1000)
    if video == None:
        return None
    video.paths.delete_dir('videoStorage')
    return None


print('\nInitialisation de la procédure\n')
video = None

try :

    # On récupère la vidéo et ses caractéristiques
    video = Video()
    interact.markerscolor_input(video)
    interact.orientation_input(video)
    interact.ref_input(video)

    # On definit les réglages par défault
    settings = Settings(video)

    # On traite la première frame  pour vérifier que les reglages sont bons
    isOK = False
    while not isOK:
        # Tant que le traitement n'est pas satisfaisant on recommence cette étape
        calibration()
        if interact.yn('Le traitement est-il bon ?'):
            isOK = True
        else:
            # lorsque le traitement n'est pas satisfaisant, il est proposé de modifier les réglages
            interact.verif_settings(video, settings)
            settings.definition, settings.step = 1, 1
            video.Frames[0].identifiedObjects = []

    # Une fois que tout est bon on traite la vidéo
    videotreatement()

    # On télécharge les données
    download.reboot(video)
    download.data(video, settings)

    if interact.yn("Voulez vous télécharger les résultats de l'étude ?"):
        download.results(video, settings.crosswidth)

    print('\nProcédure terminée')

except (Break, KeyboardInterrupt):
    cleaner(video)
    print('\n\nProcédure terminée')

# cleaner(video)
