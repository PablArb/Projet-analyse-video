from Modules import *

from Video_constructor import *
from Error_constructor import *

from Paths import *
from IHM import *
from Data_download import resultsdownload
from Video_treatement import calibration, videotreatement

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
    # delete_dir('bac')

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

main ()