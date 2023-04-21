import sys
import time as t

# modules supplémentaires
import numpy as np

# fichiers propres au projet
from Base import SettingError, Break
from Base import mess
from IHM import interact
from MainConstructor import Video, Frame, Object, Mesure, rate_rgb
from SettingsConstructor import Settings

# Main functions
def videotreatment(video: Video) -> None:
    """
    video : vidéo étudiée.
        
    Permet le traitement de l'ensemble des frames qui constituent la vidéo ainsi que le suivi des objets.
    """
    frames = video.Frames
    settings = video.settings
    mc = video.markerscolor

    print()
    Ti, T = t.time(), t.time()

    for frame in frames[1:]:  # frame 0 traitée durant l'initialisation
        try:
            frametreatement(frame, settings, mc)
            object_tracker(video, frame)
        except SettingError:
            raise Break

        if t.time() - T >= 1:
            progr = (frame.id / (len(frames) - 1)) * 100
            progr = str(round(progr))
            tleft = interact.waiting_time(frame.id, len(frames), Ti)
            print(mess.S_vtm + progr + ' % (' + tleft + ')', end='')
            T = t.time()

    d = interact.time_formater(t.time() - Ti)
    video.computationDuration = d

    print(mess.E_vtm, end='')
    print(mess.S_dvt + d, end='')

    return None

def frametreatement(frame: Frame, settings: Settings, mc: int, calib=False):
    """
    frame : image à traiter (tableau uint8).
    settings : paramètres avec lesquels la frame est traitée.
    mc : markerscolor, couleur des repères sur la frame étudiée
    
    Traite la frame passée en argument.(renvoie les postions des repères qui y sont detectés)
    """
    isOK = False
    im = frame.array
    while not isOK and settings.precision <= settings.maxPrec:
        try:
            Ti = t.time()
            mesures, Bduration = objects_detection(im, settings, mc)
            Tduration = t.time() - Ti
            isOK = True
        except RecursionError:
            print(mess.P_rec, end='')
            settings.precision += 100
            sys.setrecursionlimit(settings.precision)

    if isOK:
        Ti = t.time()
        cleanedMesures = rectifyer(mesures, settings.minsize)
        while [mes.id for mes in cleanedMesures] != [mes.id for mes in mesures]:
            mesures = cleanedMesures
            cleanedMesures = rectifyer(mesures, settings.minsize)

        Tduration += t.time() - Ti
        frame.mesures = cleanedMesures
        if calib:
            positions = [mes.pos for mes in cleanedMesures]
            borders = [mes.borders for mes in cleanedMesures]
            extremas = [mes.extremas for mes in cleanedMesures]
            return positions, borders, extremas, Bduration, Tduration
        else:
            return None

    else:
        raise SettingError


# sub functions
def objects_detection(image: np.array, settings: Settings, mc: int) -> tuple:
    """
    image : frame à traiter en N&B.
    settings : paramètres avec lesquels l'image sera traitée.
    mc : markerscolor, couleur des repères sur l'image étudiée.
    i : indice de l'image à traiter.
    
    Detecte les repères présents sur l'image passée en argument.
    """
    global at_border
    pas, tol = settings.step, settings.tol
    maxb, minb = settings.maxBrightness, settings.minBrightness
    h, w = image.shape[:2]
    mesures, id = [], 0
    s = 0

    for j in range(0, h, pas):
        for i in range(0, w, pas):

            if rate_rgb(image[j][i], mc, maxb, minb) > tol:

                # On vérifie que l'élément étudié n'appartient pas déjà à un repère détecté.
                element_in = False
                for mes in mesures:
                    extr = mes.extremas
                    HorizontalAlignement = extr[1] <= j <= extr[3]
                    VerticalAlignement = extr[0] <= i <= extr[2]
                    if VerticalAlignement and HorizontalAlignement:
                        element_in = True

                if not element_in:

                    # On initie ici la détection du contour du repère
                    depart = [i, j]
                    object = [depart]
                    init_extr = [depart[0], depart[1], depart[0], depart[1]]

                    # On considère que le premier pixel que l'on trouve n'appartiens pas forcément au contour du repère
                    at_border = False

                    Ti = t.time()
                    extremas, border = border_detection(image, depart, object, init_extr, mc, settings)
                    s += t.time() - Ti

                    mesures.append(Mesure(id, extremas, border))
                    id += 1

    return mesures, s

def border_detection(image: np.array, start: list, contour: list, extr: list, mc: int, settings: Settings) -> tuple:
    """
    image : image étudiée.
    start : pixel duquel on va partir pour 'explorer' notre objet, sous la forme [j,i].
    obj : liste contenant tous les pixels appartenants au même objet.
    extr : coordonées extremales de l'objet.
    mc : markerscolor, couleur des repères qui constituent les objets à detecter.
    tol : seuil de detection des couleurs.
    
    Regroupe tous les pixels appartenant à un même objet (forme blanche ici) dans une liste.
    """
    # On cherche ici à récupérer un encadrement de l'objet
    if start not in contour:
        contour.append(start)
        if start[0] < extr[0]:
            extr[0] = start[0]
        elif start[1] < extr[1]:
            extr[1] = start[1]
        if start[0] > extr[2]:
            extr[2] = start[0]
        elif start[1] > extr[3]:
            extr[3] = start[1]

    for pixel in get_neighbours(image, start, mc, settings):
        if pixel not in contour:
            border_detection(image, pixel, contour, extr, mc, settings)
    return extr, contour

def get_neighbours(image: np.array, pixel: list, mc: int, settings: Settings) -> list:
    """
    image : image étudiée.
    pixel : sous la forme [j,i].
    mc : markerscolor, couleur des repères sur l'image étudiée.
    
    Renvoie la liste des voisins du pixel 'pixel' à étudier dans le cadre de la recherche d'objet.
    L'idée est de fixer des conditions pour qu'un pixel face partis des voisins intéressants :
        Au départ, on n'est pas sur le contour du repère, on cherche alors à le rejoindre, on prend comme seul voisin
        le pixel à sa droite.
        Si on se trouve bien sur le contour les voisins intéressants sont les voisins se trouvant également sur le
        contour.
    """
    global at_border
    x, y = pixel[0], pixel[1]
    h, w = len(image), len(image[0])
    tol = settings.tol
    maxb, minb = settings.maxBrightness, settings.minBrightness
    view = settings.view

    # On crée une liste des coordonnées des voisins potentiellement intéressants
    neighbours_coordinates = []
    for i in range(-view, view + 1):
        for j in range(-view, view + 1):
            if j != i:
                neighbours_coordinates.append(((x + i) % w, (y + j) % h))

    # On crée la liste des voisins n'appartenant âs au repère.
    # On vérifie en même temps si on a atteint le contour du repère ou non.
    is_border = False
    outsiders = []
    for n in neighbours_coordinates:
        if rate_rgb(image[n[1], n[0]], mc, maxb, minb) < tol:
            is_border = True
            outsiders.append(n)
            # Si on n'était pas sur le contour, on y est désormais.
            if not at_border:
                at_border = True

    L_neighbours = []
    # Si on ne se trouve pas sur le contour du repère, on prend comme voisin seulement le pixel de droite.
    if not is_border and not at_border:
        L_neighbours.append((pixel[0] + 1, pixel[1]))
    # Sinon on ne garde que les éléments proches d'éléments qui n'appartiennent pas au repère.
    if is_border:
        for n in neighbours_coordinates:
            if n not in outsiders:
                for o in outsiders:
                    if abs(n[0] - o[0]) <= 1 and abs(n[1] - o[1]) <= 1:
                        L_neighbours.append(n)

    return L_neighbours

def rectifyer(mesures: list[Mesure], minsize: int) -> list[Mesure]:
    """
    mesures : liste contenats les objets détectés sur la frmae étudiée
    minsize : taille minimum que doit avoir un objet pour ne pas être considéré comme du bruit.

    Dans un premier temps, on regroupe les mesures liées à un même repère (elles peuvent être séparée si séparée
    d'un unique pixel) puis on supprime les mesures considérées comme du bruit.
    """

    # d = max([max(mes.size) for mes in mesures])

    newMes1 = []
    dictRectified = {mes: False for mes in mesures}

    while not all([dictRectified[mes] for mes in mesures]):
        notRectified = [mes for mes in mesures if not dictRectified[mes]]
        mes1 = notRectified[0]
        d = max(mes1.size)
        dictRectified[mes1] = True
        group = [mes1]

        for mes2 in notRectified:
            if mes2 != mes1:
                d2 = distance(mes1.pos, mes2.pos)
                if d2 <= d:
                    group.append(mes2)
                    dictRectified[mes2] = True

        extr = np.array([mes.extremas for mes in group], dtype=object)
        xmin, ymin = min(extr[:, 0]), min(extr[:, 1])
        xmax, ymax = max(extr[:, 2]), max(extr[:, 3])

        borders = []
        for mes in group:
            borders += mes.borders

        newMes1.append(Mesure(mes1.id, (xmin, ymin, xmax, ymax), borders))

    newMes2 = []
    for mes in newMes1:
        if not (mes.size[0] < minsize or mes.size[1] < minsize):
            newMes2.append(mes)

    return newMes2


# Tracking functions
def object_tracker(video: Video, frame: Frame) -> None:
    """
    video : vidéo étudiée
    frame : frame étudiée

    Effectue le suivi des repères d'une frame à la suivante.
    """
    markers = video.markers
    mesures = frame.mesures
    maxdist = video.settings.maxdist
    M = np.zeros((len(markers), len(frame.mesures)))

    for i in range(len(markers)):
        obj = markers[i]
        if obj.status == 'hooked':

            obj.lastupdate += 1
            pred = obj.kf.predict()
            obj.predictions[frame.id] = pred

            dist = distances(pred, mesures, maxdist)
            if len(dist) != 0:
                ind = np.argmin(dist)
                if dist[ind] < np.inf:
                    M[i][ind] = 1

    # M : colonnes = mesures, lignes = markers
    # Ainsi si la somme sur la colonne n'est pas égale à 1 cela signifie que différents markers pourrait se trouver
    # à cette position. Si elle est bien égale à 1 il n'y a pas de conflit.
    res = np.sum(M, axis=0)

    matching(frame, M, res, mesures, markers)

    for obj in markers:
        if obj.status == 'hooked':
            if obj.lastupdate != 0:
                video.treatementEvents += f'frame {frame.id}\tobject not found\t{obj.id}\n'
                obj.positions[frame.id] = [int(pred[0]), int(pred[1])]
            if obj.lastupdate >= 5:
                obj.status = 'lost'
                video.treatementEvents += f'frame {frame.id}\tobject lost\t{obj.id}\n'

        elif obj.status == 'lost':
            obj.positions[frame.id] = obj.lastknownpos
    return None

def distances(pred: list, mesures: list[Mesure], maxdist: int) -> list:
    """
    pred : position prédite [x, y]
    mesures : liste de positions des objets détéctés

    Calcule les distances entre la position prédite et les positions de chacuns des objets détéctés.
    """
    LDistances = []
    for mes in mesures:
        d = distance(pred, mes.pos)
        if d > maxdist:
            d = np.inf
        LDistances.append(d)
    return LDistances

def distance(p1: tuple, p2: tuple) -> float:
    return np.sqrt((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) ** 2)

def matching(frame: Frame, M: np.array, res: np.array, mesures: list, markers: list) -> None:
    # On interprète les résultats
    imatchMesures = [i for i in range(len(res)) if res[i] == 1]
    imatchMarkers = [np.argmax(M[:, i]) for i in imatchMesures]
    matchedMesures = [mesures[i] for i in imatchMesures]
    matchedMarkers = [markers[i] for i in imatchMarkers]

    assert len(matchedMesures) == len(matchedMarkers)
    for i in range(len(matchedMarkers)):
        update(frame, matchedMarkers[i], matchedMesures[i])

    return None

def update(frame: Frame, marker: Object, mesure: Mesure) -> None:
    mesure.status = 'matched'
    marker.positions[frame.id] = mesure.pos
    marker.lastknownpos = mesure.pos
    marker.kf.update(np.expand_dims(mesure.pos, axis=-1))
    marker.lastupdate = 0
    frame.identifiedObjects.append(marker)
    return None
