import sys

from Base import SettingError
from Base import mess
from Base import time_formater
from IHM import visu, interact
from Constructor import Video, Object
from VideoTreatment import frametreatement


def calibration(video: Video, i=0) -> None:
    """
    video : vidéo à traiter.

    Permet de vérifier le bon réglage de l'ensemble des paramètres.
    """
    # On va dans un premier temps traiter la première frame de la video.
    print(mess.B_cal, end='')

    settings = video.settings
    first = video.Frames[i]
    mc = video.markerscolor

    try:
        positions, borders, extremas, Bdur, Tdur = frametreatement(first, settings, mc, True)
    # On n'est pas assuré de la capacité de l'algorithme à traiter l'image avec les paramètres entrés par
    # l'utilisateur, on gère ici ce problème.
    except SettingError:
        interact.verif_settings(video)
        reboot(video)
        calibration(video)
        return None

    detPas(video, extremas)
    detScale(video, positions)

    swipDur = Tdur - Bdur  # durée nécessaire au balayage de chaque image
    videoDur = (swipDur / (settings.step ** 2) + Bdur) * len(video.Frames)
    formatedDur = time_formater(videoDur)

    # Une fois le traitement réalisé on stocke les résultats.
    video.markercount = 0
    for obj in positions:
        new_obj = Object('obj-' + str(video.markercount), obj, first.id)
        first.identifiedObjects.append(new_obj)
        video.markers.append(new_obj)
        video.markercount += 1

    print(mess.E_cal, end='')

    # On crée maintenant les visuels à partir des résultats.
    visu.visus(video, first, borders, extremas)

    print(mess.S_dur + str(formatedDur), end='')
    return None


def detPas(video: Video, extr: dict) -> None:
    """
    video : vidéo étudiée.
    extr : {0: [xmin, ymin, xmax, ymax], 1: ... },
        dictionaire où chaque clef correspond à un objet,
        la valeure qui lui est associée est la liste des 4 coordonées
        extremales entourant l'objet.

        Associe à l'attribut step des reglages de la vidéo l'intervalle le
    plus large tel que l'étude reste faisable.
    """
    if len(extr) == 0:
        return None
    mini = min(extr[0][2] - extr[0][0], extr[0][3] - extr[0][1])
    for el in extr:
        if el[2] - el[0] < mini:
            mini = el[2] - el[0]
        if el[3] - el[1] < mini:
            mini = el[3] - el[1]
    video.settings.step = mini // 2
    # On multiplie par 3 pour s'assurer de ne manquer aucun repère.
    return None


def detScale(video: Video, positions: dict) -> None:
    """
    positions : dictionaire contenant les positions de chaque repère sur une des frames.
    lenref : longeur de reférance sur laquelle on s'appuie pour définir l'échelle.

    Renvoie l'échelle de la vidéo en cm par nb de pixel.
    """
    lenref = video.lenref
    if len(positions) >= 2:
        a = positions[-1]
        b = positions[-2]
        xa, ya, xb, yb = a[0], a[1], b[0], b[1]
        scale = lenref / (((xa - xb) ** 2 + (ya - yb) ** 2) ** (1 / 2))

    else:
        scale = 1
    video.scale = scale
    return None


def reboot(video: Video, i=0) -> None:
    video.settings.precision = 1000
    sys.setrecursionlimit(1000)
    video.settings.step = 1
    video.Frames[i].identifiedObjects = []
    video.markers = []
    return None
