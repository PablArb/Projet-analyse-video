#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:04:06 2022

@author: pabloarb
"""
import shutil as sht
import sys

from Base import Break, mess
from IHM import download, interact
from MainConstructor import Video
from Calibration import calibration, reboot
from VideoTreatment import videotreatment


def cleaner(video: Video, isOK=True) -> None:
    """
    video : Video, video que l'on souhaite traiter
    isOK : booléen, optionel, si le traitement n'est pas terminé, mais que l'utilisateur veut interompre l'algorithme,
        sa video sera copiée sur le bureau.

    Efface les traces que laisse l'algorithme sur loridnateur de l'utilisateur.
    """
    sys.setrecursionlimit(1000)
    if not isOK:
        dst = video.paths.desktop
        src = video.paths.videoStorage + '/' + video.id
        sht.copy2(src, dst)
    video.paths.delete_dir('videoStorage')
    video.paths.delete_dir('bac')
    return None


print(mess.B_proc, end='')
video = None

try:

    # On récupère la vidéo et ses caractéristiques
    video = Video()
    interact.markerscolor_input(video)
    interact.orientation_input(video)
    interact.ref_input(video)

    # On traite la première frame pour vérifier que les réglages sont bons
    isOK = False
    while not isOK:
        # Tant que le traitement n'est pas satisfaisant on recommence cette étape
        calibration(video)
        if interact.yn(mess.I_val):
            isOK = True
        else:
            # lorsque le traitement n'est pas satisfaisant, il est proposé de modifier les paramètres.
            interact.verif_settings(video)
            reboot(video)
    # Une fois que tout est bon on traite la vidéo.
    videotreatment(video)
    # On télécharge les résultats.
    download.reboot(video)
    download.data(video)

    if interact.yn(mess.I_dlr):
        download.results(video)
    cleaner(video)
    print(mess.E_proc)

except (Break, KeyboardInterrupt):
    cleaner(video, isOK=False)
    print(mess.E_proc)
