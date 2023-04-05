#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:04:06 2022

@author: pabloarb
"""
import shutil as sht
import sys

from Base import Break, SettingError, mess
from IHM import visu, download, interact
from VideoTreatment import Video, Object
from VideoTreatment import calib, time_formater
from VideoTreatment import videotreatment, frametreatement


# Calibration fcts

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
        calib.reboot(video)
        calibration(video)
        return None

    calib.detPas(video, extremas)
    calib.detScale(video, positions)

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
            calib.reboot(video)
    # Une fois que tout est bon on traite la vidéo.
    videotreatment(video)
    # On télécharge les résultats.
    download.reboot(video)
    download.data(video)

    if interact.yn("Voulez vous télécharger les résultats de l'étude ?"):
        download.results(video)
    cleaner(video)
    print(mess.E_proc)

except (Break, KeyboardInterrupt):
    cleaner(video, isOK=False)
    print(mess.E_proc)
