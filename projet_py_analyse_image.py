import cv2
import sys
import time as t
from IHM import visu, download, interact
from ERRORS import Break, SettingError
from VideoTreatment import Video, Object
from VideoTreatment import calib, position, reducer
from VideoTreatment import videotreatement, frametreatement


# Calibration fcts

def calibration(video):
    """
    À effectuer avant le traitement de l'ensemble de la vidéo pour vérifier le
    bon réglage de l'ensmeble des paramètres.
    """
    settings = video.settings
    print()
    print('Traitement en cours ...', end='')

    try :

        detected = frametreatement(video.Frames[0].array, settings, video.markerscolor, 0)
        extremums = detected[0]
        positions = position(extremums)
        
        calib.detPas(video, extremums, settings.definition)
        calib.detScale(video, video.lenref, positions)

    except SettingError :
        print('\rIl y a un problème, veuillez vérifiez les réglages', end='\n' )
        interact.verif_settings(video, settings)
        video.settings.definition = 1
        video.settings.step = 1
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

    rectanglewidth = video.settings.rectanglewidth
    crosswidth = video.settings.crosswidth
    markerscolor =  video.markerscolor
    scale = video.scale
    first = video.Frames[0]

    print('Création des visuels en cours ...', end='')
    visualisations = []

    color_im = visu.copy_im(first.array)
    visualisations.append(color_im)

    NB_im = reducer(color_im, settings.definition)
    NB_im = visu.reduced(video, settings, NB_im)
    NB_im = cv2.resize(NB_im, video.Framessize)
    visualisations.append(NB_im)

    treated_NB = visu.copy_im(NB_im)
    treated_NB = visu.detection(treated_NB, detected[1])
    treated_NB = visu.rectangle_NB(treated_NB, extremums, rectanglewidth)
    visualisations.append(treated_NB)

    pos = [obj.positions[first.id] for obj in first.identifiedObjects]
    treated_color = visu.cross_color(first.array, pos, crosswidth, copy=True)
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

    # On traite la première frame  pour vérifier que les reglages sont bons
    isOK = False
    while not isOK:
        # Tant que le traitement n'est pas satisfaisant on recommence cette étape
        calibration(video)
        if interact.yn('Le traitement est-il bon ?'):
            isOK = True
        else:
            # lorsque le traitement n'est pas satisfaisant, il est proposé de modifier les réglages
            interact.verif_settings(video)
            video.settings.definition, video.settings.step = 1, 1
            video.Frames[0].identifiedObjects = []

    # Une fois que tout est bon on traite la vidéo
    videotreatement(video)

    # On télécharge les données
    download.reboot(video)
    download.data(video)

    if interact.yn("Voulez vous télécharger les résultats de l'étude ?"):
        download.results(video)

    print('\nProcédure terminée')

except (Break, KeyboardInterrupt):
    cleaner(video)
    print('\n\nProcédure terminée')

# cleaner(video)
