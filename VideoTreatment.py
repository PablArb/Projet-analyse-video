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
    mc : markerscolor, couleur des repères sur la frame étudiée.
    i : numméro de la frame que l'on traite.
    
    Traite la frame passée en argument.(renvoie les postions des repères qui y sont detectés)
    """
    isOK = False
    im = frame.array
    while not isOK and settings.precision <= settings.maxPrec:
        try:
            Ti = t.time()
            extremas, borders, Bduration = objects_detection(im, settings, mc)
            Tduration = t.time() - Ti
            isOK = True
        except RecursionError:
            print(mess.P_rec, end='')
            settings.precision += 100
            sys.setrecursionlimit(settings.precision)

    if isOK:
        extremas = rectifyer(extremas, settings.minsize)
        positions = position(extremas)
        frame.mesures = [Mesure(pos) for pos in positions]

        if calib:
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
    h, w = image.shape[:2]
    extremas, borders = [], []
    s = 0

    for j in range(0, h, pas):
        for i in range(0, w, pas):

            if rate_rgb(image[j][i], mc) > tol:

                # On vérifie que l'élément étudié n'appartient pas déjà à un repère détecté.
                element_in = False
                for obj in extremas:
                    HorizontalAlignement = obj[1] <= j <= obj[3]
                    VerticalAlignement = obj[0] <= i <= obj[2]
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
                    res = border_detection(image, depart, object, init_extr, mc, tol)
                    s += t.time() - Ti

                    extremas.append(res[0])
                    borders.append(res[1])

    return extremas, borders, s

def border_detection(image: np.array, start: list, obj: list, extr: list, mc: int, tol: float) -> tuple:
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
    if start not in obj:
        obj.append(start)
        if start[0] < extr[0]:
            extr[0] = start[0]
        elif start[1] < extr[1]:
            extr[1] = start[1]
        if start[0] > extr[2]:
            extr[2] = start[0]
        elif start[1] > extr[3]:
            extr[3] = start[1]

    for pixel in get_neighbours(image, start, mc, tol):
        if pixel not in obj:
            border_detection(image, pixel, obj, extr, mc, tol)
    return np.array(extr), np.array(obj)

def get_neighbours(image: np.array, pixel: list, mc: int, tol: float) -> list:
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
    h = len(image)
    w = len(image[0])
    view = 2

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
        if rate_rgb(image[n[1], n[0]], mc) < tol:
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

def rectifyer(extremas: dict, minsize: int) -> list:
    """
    extremas : dictionaire contenant les coordonnées extremales des repères détectés sur une frame.
    minsize : Taille minimale acceptée pour un objet.
    
    Rectifie quelques erreurs, élimine le bruit.
    """
    # On supprime les objets trop petits, probablement issus d'erreurs.
    new_extremas = []

    for i in range(len(extremas)):
        obj = extremas[i]
        if not (obj[2] - obj[0] < minsize or obj[3] - obj[1] < minsize):
            new_extremas.append(obj)

    return new_extremas

def position(extremas: list) -> list:
    """
    extremas : dictionaire contenant les coordonnées extremales des repères détectés sur une frame.
    
    Détermine la position d'un objet à partir des extremas.
    Renvoie un dictionnaire où les clefs sont les noms des différents objets détectés sur la frame étudiée et les
    valeurs sont les coordonées du 'centre' de l'objet.
    """
    position = []
    for obj in extremas:
        x = (obj[0] + obj[2]) / 2
        y = (obj[1] + obj[3]) / 2
        position.append([x, y])
    return position


# Tracking functions
def object_tracker(video: Video, frame: Frame) -> None:
    """
    video : vidéo étudiée
    frame : frame étudiée

    Effectue le suivi des repères d'une frame à la suivante.
    """
    markers = video.markers
    mesures = frame.mesures
    M = np.zeros((len(markers), len(frame.mesures)))

    for i in range(len(markers)):
        obj = markers[i]
        if obj.status == 'hooked':

            obj.lastupdate += 1
            pred = obj.kf.predict()
            obj.predictions[frame.id] = pred

            dist = distances(pred, mesures)
            if len(dist) != 0:
                M[i][np.argmin(dist)] = 1

    # M : colonnes = mesures, lignes = markers
    # Ainsi si la somme sur la colonne n'est pas égale à 1 cela signifie que différents markers pourrait se trouver
    # à cette position. Si elle est bien égale à 1 il n'y a pas de conflit.
    res = np.sum(M, axis=0)

    matching(frame, M, res, mesures, markers)

    for obj in markers:
        if obj.status == 'hooked':
            if obj.lastupdate != 0:
                video.treatementEvents += f'frame {frame.id}\t\tobject not found\t{obj.id}\n'
                obj.positions[frame.id] = [int(pred[0]), int(pred[1])]
                print('!!!!!!!!!!', end='')
            if obj.lastupdate >= 5:
                obj.status = 'lost'
                video.treatementEvents += f'frame {frame.id}\t\tobject lost\t\t{obj.id}\n'

        elif obj.status == 'lost':
            obj.positions[frame.id] = obj.lastknownpos
    return None

def distances(pred: list, mesures: list[Mesure]) -> list:
    """
    pred : position prédite [x, y]
    mesures : liste de positions des objets détéctés

    Calcule les distances entre la position prédite et les positions de chacuns des objets détéctés.
    """
    LDistances = []
    xp, yp = int(pred[0]), int(pred[1])
    for mes in mesures:
        xm, ym = mes.pos[0], mes.pos[1]
        d = ((xp - xm) ** 2 + (yp - ym) ** 2) ** .5
        LDistances.append(d)
    return LDistances

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
