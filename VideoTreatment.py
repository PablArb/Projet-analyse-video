#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 16:32:07 2022

@author: pabloarb
"""

import sys, os, cv2
import numpy as np
import time as t
import shutil as sht
import pymediainfo as mi
from Base import paths, SettingError



class Settings:
    def __init__(self, video):

        self.precision = 200       # permet de gérer la precision du système
        self.tol = 0.4             # est réglable lors de l'execution
        self.maxdef = 15           # abaissement de la definition maximal
        self.definition = 1        # est automatiquement réglé par le programme
        self.step = 1              # est automatiquement réglé par le programme
        sys.setrecursionlimit(self.precision)

        # On définit la taille des indicateurs visuels / taille de l'image
        self.minsize = int(video.Framessize[1] / 170)
        self.maxdist = int(video.Framessize[1] / video.Framerate * 10)
        self.bordure_size = 0
        # self.bordure_size = int(video.Framessize[0] /  video.Framerate * 2)
        self.crosswidth = int(video.Framessize[1] / 500)
        self.rectanglewidth = int(video.Framessize[1] / 1250)


class Video(object):

    def __init__(self):

        self.paths = paths
        
        self.id = None #titre de la vidéo 
        self.videoinput()

        self.Frames = self.get_frames() #liste contenant les frames de la vidéo
        self.Framerate = self.get_framerate() # nombre de frame par seconde
        self.Framessize = self.get_framessize() # taille des frames

        self.markerscolor = None #couleur des repères visuels sur la video
        self.orientation = None #orientation de la video (paysage ou portrait)
        self.lenref = None #longueur de référence associée à la video

        self.scale = None #rapport distance sur nombre de pixel
        self.markercount = 0 #nombre de repères détectés sur la vidéo
        self.computationDuration = None #temps mis par l'algorythme pour 
                                        #effectuer le traitement
        
        self.settings = Settings(self) #réglages associés à la vidéo
        
        
    def videoinput(self) -> None:
        '''
        Pas d'argument.
        
        Récupère la vidéo auprès de l'utilisateur.
        '''
        self.paths.create_dir('bac')
        isempty = True
        print('\rPlacez la vidéo à étudier dans le bac sur votre bureau.', end='')
        while isempty:
            if len(os.listdir(self.paths.bac)) != 0:
                isempty = False
            t.sleep(0.5)
        _bac = os.listdir(self.paths.bac)
        ext = _bac[0].split('.')[1]
        if len(_bac) == 1 and (ext == 'mp4' or ext == 'mov'):
            video = _bac[0]
            self.paths.vidéoinput = self.paths.bac + '/' + video
            self.paths.create_dir('videoStorage')
            sht.copy2(self.paths.vidéoinput, self.paths.videoStorage)
            self.id = str(video)
            self.paths.delete_dir('bac')
            return None
        elif len(_bac) == 1 and ext != 'mp4' and ext != 'mov' :
            print('\rVeuillez fournir une vidéo au format mp4')
            source = self.paths.bac + '/' + _bac[0]
            destination = self.paths.desktop + '/' + _bac[0]
            sht.copy2(source, destination)
            self.paths.delete_dir('bac')
            self.videoinput()
        elif len(_bac) > 1:
            print("\rVeuillez ne placer qu'un document dans le bac", end='')
            self.paths.delete_dir('bac')
            self.videoinput()

    def get_frames(self) -> list:
        """
        Pas d'argument.
        
            Renvoie une liste contenant l'ensemble des frames (tableaux de type
        uint8) dans le même ordre que dans la vidéo étudiée.
        """
        frames = []
        cam = cv2.VideoCapture(self.paths.videoStorage + '/' + self.id)
        frame_number = 0
        print('\rRécupération de la vidéo en cours ...', end='')
        while True:
            ret, frame = cam.read()
            if ret:
                frames.append(Frame('frame.' + str(frame_number), frame))
                frame_number += 1
            else:
                break
        cam.release()
        cv2.destroyAllWindows()
        print('\rRécupération de la vidéo ------------------------------------------ OK', end='\n\n')
        t.sleep(0.1)
        return frames

    def get_framerate(self) -> float:
        """
        Pas d'argument.
        
        Renvoie le nombre de frames par secondes de la vidéo étudiée.
        """
        media_info = mi.MediaInfo.parse(self.paths.videoStorage + '/' + self.id)
        tracks = media_info.tracks
        for i in tracks:
            if i.track_type == 'Video':
                framerate = float(i.frame_rate)
        return framerate

    def get_framessize(self) -> tuple:
        """
        Pas d'argument.
        
            Renvoie un tuple de deux valeurs : la hauteur et largeur des frames
        de la vidéo.
        """
        media_info=mi.MediaInfo.parse(self.paths.videoStorage + '/'+self.id)
        video_tracks = media_info.video_tracks[0]
        w = int(video_tracks.sampled_width)
        h = int(video_tracks.sampled_height)
        framessize = (w, h)
        return framessize

class Frame:
    def __init__(self, id, array):
        self.id = id
        self.array = array
        self.AreasOfInterest = []
        self.identifiedObjects = []

class Object:
    def __init__(self, id):
        self.id = id
        self.status = 'in'
        self.positions = {}
        self.LastKnownPos = None

class Calib:
    
    def detPas (self, video:Video, extr:dict) -> None:
        '''
        video   : vidéo étudiée.
        extr    : {0: [xmin, ymin, xmax, ymax], 1: ... },
            dictionaire où chaque clef correspond à un objet,
            la valeure qui lui est associée est la liste des 4 coordonées
            extremales entourant l'objet.
            
            Associe à l'attribut step des reglages de la vidéo l'intervalle le 
        plus large tel que l'étude reste faisable.
        '''
        definition = video.settings.definition
        L = list(extr.keys())
        if len(L) == 0:
            return 1
        min = extr[L[0]][2] - extr[L[0]][0]
        for el in extr :
            if extr[el][2] - extr[el][0] < min :
                min = extr[el][2] - extr[el][0]
            if extr[el][3] - extr[el][1] < min :
                min = extr[el][3] - extr[el][1]
        video.settings.step = int(min/(definition * 4)) # On multiplie par 3 pour s'assurer de ne manquer aucun repère.
        return None


    def detScale (self, video:Video, positions:dict) -> float:
        '''
        positions   : dictionaire contenant les positions de chaque repère sur
            une des frames.
        lenref      : longeur de reférance sur laquelle on s'appuie pour 
            définir l'échelle.

        Renvoie l'échelle de la vidéo en cm par nb de pixel.
        '''
        lenref = video.lenref
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
    
    def reboot(self, video:Video, i:int)-> None :
        video.settings.definition = 1
        video.settings.step = 1
        video.Frames[i].identifiedObjects = []
        return None
    
# Traitement tools

def videotreatment(video:Video) -> None:
    """
    video : vidéo étudiée.
        
        Permet le traitement de l'ensemble des frames qui constituent la vidéo 
    ainsi que le suivi des objets.
    """
    frames = video.Frames
    settings = video.settings
    mc = video.markerscolor
    maxdist = settings.maxdist
    bordure_size = settings.bordure_size
    

    Ti, T = t.time(), t.time()
    print()

    for i in range(1, len(frames)): # frame 0 traitée durant l'initialisation
        try :

            markers_extr = frametreatement(frames[i].array, settings, mc, i)[0]
            positions = position(markers_extr)

            object_tracker(video, i, positions, maxdist, bordure_size)

        except SettingError :
            print('\rproblèmes dans les réglages')

        if t.time() - T >= 1 :
            progr = (int(frames[i].id.split('.')[1]) / (len(frames) - 1)) * 100
            progr = str(round(progr))
            tleft = waiting_time(i, len(frames), Ti)
            print('\033[2K\033[1GTraitement en cours : ' +progr+ ' % (' +tleft+ ')', end='')
            T = t.time()
            
    name = ''.join( tuple( video.id.split('́') ) )
    print('\rTraitement de ' + name + ' ' + '-'*( 82-len(name) ) + ' OK', end='\n\n')

    video.computationDuration = round(t.time()-Ti, 1)

    return None

def frametreatement(frame:np.array, settings:Settings, mc:int, i:int) -> tuple:
    """
    frame       : image à traiter (tableau uint8).
    settings    : paramètres avec lesquels la frame est traitée.
    mc          : markerscolor, couleur des repères sur la frame étudiée.
    i           : numméro de la frame que l'on traite.
    
        Traite la frame passée en argument.(renvoie les postions des repères 
    qui y sont detectés)
    """
    isOK = False
    
    while not isOK and settings.definition <= settings.maxdef :
        try:
            image = reducer(frame, settings.definition)
            extremas, borders = objects_identification(image, settings, mc, i)
            isOK = True
        except RecursionError:
            print('\rDéfinition trop élevée, tentative avec une défintion plus faible', end='')
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
        
        extremas = rectifyer(extremas, settings.minsize)
        return extremas, borders
    else:
        raise SettingError

def reducer(image:np.array, definition:int) -> np.array:
    """
    image       : image de depart.
    Definition  : l'image finale contiendra 1/definition² pixels de l'image
        initiale.
        
    Réduit la l'image afin de réduire la quantité d'information à traiter.
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

def objects_identification(image:np.array, settings:Settings, mc:int, i:int) -> tuple :
    """
    image       : frame à traiter.
    settings    : paramètres avec lesquels l'image sera traitée.
    mc          : markerscolor, couleur des repères sur l'image étudiée.
    i           : indice de l'image à traiter.
    
    Detecte les repères présents sur l'image passée en argument.
    """
    global at_border
    pas, tol = settings.step, settings.tol
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

            if not element_in and rate_rgb(image[j][i], mc) > tol :
                depart = [i, j]
                object = [depart]
                init_extr = [depart[0], depart[1], depart[0], depart[1]]
                at_border = False
                extremas[n], borders[n] = detection(image, depart, object, init_extr, mc, tol)
                n += 1

    return extremas, borders

def rate_rgb(pixel:list, c:int) -> float:
    """
    pixel : élement de l'image d'origine sous la forme [r, g, b].
        c = 0(rouge), 1(vert) ou 2(bleu).

        Calcul le poids relatif de la composante c du pixel parmis les
    composantes rgb qui le définissent.
    """
    assert c in [0, 1, 2]
    return int(pixel[c]) / (int(pixel[0]) + int(pixel[1]) + int(pixel[2]) + 1)

def detection(image:np.array, start:list, obj:list, extr:list, mc:int, tol:float) -> list:
    """
    image   : image étudiée.
    start   : pixel duquel on va partir pour 'explorer' notre objet, 
                sous la forme [j,i].
    obj     : liste contenant tout les pixels appartenants au même objet.
    extr    : coordonées extremales de l'objet.
    mc      : markerscolor, couleur des repères qui constituent les objets à detecter.
    tol     : seuil de detection des couleurs. 
    
        Regroupe tous les pixels appartenant a un même objets (forme blanche 
    ici) dans une liste.
    """
    if start not in obj:            # but: récupérer un encadrement de objet
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
            detection(image, pixel, obj, extr, mc, tol)
    return extr, obj


def get_neighbours(image:np.array, pixel:list, mc:int, tol:float) -> list:
    """
    image   : image étudiée.
    pixel   : sous la forme [j,i].
    mc      : markerscolor, couleur des repères sur l'image étudiée.
    
        Renvoie la liste des voisins du pixel 'pixel' à étudier dans le cadre 
    de la recherche d'objet.
    """
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
        if rate_rgb(image[n[1]][n[0]], mc) < tol :
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

def rectifyer(extremas:dict, minsize:int) -> dict:
    """
    extremas    : dictionaire contenant les coordonnées extremales des repères 
        détectés sur une frame.
    minsize     : Taille minimale acceptée pour un objet.
    
    Rectifie quelques erreurs, élimine le bruit.
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

def position(extremas:dict) -> list:
    """
    extremas    : dictionaire contenant les coordonnées extremales des repères 
        détectés sur une frame.
    
        Détermine la position d'un objet à partir des extremas.
    Renvoie un dictionnaire où les clefs sont les noms des différents objets
    détectés sur la frame étudiée et les valeurs sont les coordonées du
    'centre' de l'objet.
    """
    position = {}
    for obj in extremas:
        x = (extremas[obj][0] + extremas[obj][2]) / 2
        y = (extremas[obj][1] + extremas[obj][3]) / 2
        position[obj] = [x, y]
    return position

def object_tracker(video, i, positions, maxdist, bordure_size):
    '''
    video           : vidéo étudiée.
    i               : indice de la frame étudiée dans la liste contenant
        l'ensemble des frames de la vidéo.
    maxdist         : distance à partir de laquelle un objet ayant parcouru 
        cette distance d'une frame à la suivante n'est pas considérer comme un 
        même objet.
    bordure_size    :  largeure des bordure autor de la frame permettant de 
        detecter les repères entrant dans le champ de la caméra.
        
    Effectue le suivi des repère d'une frame à la suivante.
    '''
    frames = video.Frames
    framessize = video.Framessize
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
            if in_bordure(framessize, bordure_size, positions[obj1]):
                new_obj = Object('obj-'+str(video.markercount))
                new_obj.positions[video.Frames[i].id] = [x1, y1]
                video.Frames[i].identifiedObjects.append(new_obj)
                video.markercount += 1

        for obj in video.Frames[i-1].identifiedObjects:
            if not obj in video.Frames[i].identifiedObjects:
                obj.status = 'out'
                obj.LastKnownPos = obj.positions[frames[i-1].id]

def in_bordure (framessize, bordure_size, pos):
    # Les objets apparaissant aux bordures de l'écran ne seront pas considérés
    # comme des erreurs mais comme des nouveaux objets entrant dans le chant de
    # la caméra.

    BandeGaucheHaut = [i for i in range(0, bordure_size + 1)]
    BandeBas = [i for i in range(framessize[1] - bordure_size, framessize[1] + 1)]
    BandeDroite = [i for i in range(framessize[0] - bordure_size, framessize[0] + 1)]

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


calib = Calib()
