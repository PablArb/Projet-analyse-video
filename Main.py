#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:04:06 2022

@author: pabloarb
"""
import cv2
import sys
import time as t
import shutil as sht
from IHM import visu, download, interact
from Base import Break, SettingError, mess
from VideoTreatment import Video, Object
from VideoTreatment import calib, time_formater
from VideoTreatment import videotreatment, frametreatement


# Calibration fcts

def calibration(video:Video, i=0) -> None:
    """
    video : vidéo à traiter.
    
    Permet de vérifier le bon réglage de l'ensemble des paramètres.
    """
    # On va dans un premier temps traiter le première frame de la video.
    print(mess.B_cal, end='')

    settings = video.settings
    first = video.Frames[i]
    mc = video.markerscolor
    
    try :
        T = t.time()
        positions,borders,extremums= frametreatement(first, settings, mc, True)
        duration = (t.time()-T)*len(video.Frames)
        
        calib.detPas(video, extremums)
        calib.detScale(video, positions)
        
        adaptedDur = duration/(settings.step*2)
        formatedDur = time_formater(adaptedDur)
    # On n'est pas assuré de la capacité de l'algorithme à traiter l'image avec
    # les paramètres entrés par l'utilisateur, on gère ici ce problème.
    except SettingError :
        interact.verif_settings(video)
        calib.reboot(video, i)
        calibration(video)
        return None
    
    # Une fois le traitement réalisé on stocke les résultats.
    video.markercount = 0
    for obj in positions :
        new_obj = Object('obj-'+str(video.markercount), obj, first.id, video.Framerate)
        first.identifiedObjects.append(new_obj)
        video.markers.append(new_obj)
        video.markercount += 1

    print(mess.E_cal, end='')
    
    # On créer maintenant les visuels à partir des résultats.
    rw = video.settings.rectanglewidth
    cw = video.settings.crosswidth
    scale = video.scale
    tol = settings.tol
    definition = settings.definition

    print(mess.B_vis, end='')
    visualisations = []

    color_im = visu.copy_im(first.array)
    visualisations.append(color_im)

    NB_im = visu.reduced(mc, tol, definition, color_im)
    NB_im = cv2.resize(NB_im, video.Framessize)
    visualisations.append(NB_im)

    treated_NB = visu.detection(NB_im, borders, copy=True)
    treated_NB = visu.rectangle_NB(treated_NB, extremums, rw)
    visualisations.append(treated_NB)

    pos = [obj.positions[first.id] for obj in first.identifiedObjects]
    treated_color = visu.cross_color(first.array, pos, cw, copy=True)
    treated_color = visu.scale(treated_color, scale, cw, mc)
    visualisations.append(treated_color)

    print(mess.S_vis, end='')
    
    # On présente les résultats à l'utilisateur.
    for im in visualisations :
        cv2.imshow('calibration window', im)
        cv2.waitKey(0)
        cv2.destroyWindow('calibration window')
        cv2.waitKey(1)

    print(mess.E_vis, end='')
    print(mess.S_dur + str(formatedDur), end='')
    return None


def cleaner(video:Video, isOK=True) -> None:
    '''
    video : Video, video que l'on souhaite traiter
    isOK : booléen, optionel, si le traitement n'est pas on mais que
        l'utilisateur veut interompre l'algorithme, sa video sera copiée sur 
        son bureau
        
    Efface les traces que laisse l'algorithme sur loridnateur de l'utilisateur.
    '''
    sys.setrecursionlimit(1000)
    if video == None:
        return None
    if not isOK :
        dst = video.paths.desktop
        src = video.paths.videoStorage + '/' + video.id
        sht.copy2(src, dst)
    video.paths.delete_dir('videoStorage')
    video.paths.delete_dir('bac')
    return None


print(mess.B_proc, end='')

video = None

try :

    # On récupère la vidéo et ses caractéristiques
    video = Video()
    interact.markerscolor_input(video)
    interact.orientation_input(video)
    interact.ref_input(video)

    # On traite la première frame  pour vérifier que les réglages sont bons
    isOK = False
    while not isOK:
        # Tant que le traitement n'est pas satisfaisant on recommence cette étape
        calibration(video, 0)
        if interact.yn(mess.I_val): 
            isOK = True
        else:
            # lorsque le traitement n'est pas satisfaisant, il est proposé de modifier les paramètres.
            interact.verif_settings(video)
            calib.reboot(video, 0)

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
