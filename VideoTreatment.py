#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 16:32:07 2022

@author: pabloarb
"""
import os
import shutil as sht
import sys
import time as t

import cv2
import numpy as np
import pymediainfo as mi

from Base import paths, mess, SettingError, Break


class Settings(object):
    def __init__(self, video):
        self.precision = 1000  # permet de gérer la precision du système
        self.maxPrec = 1e6
        self.tol = 40.0  # est réglable lors de l'execution
        self.step = 1  # est automatiquement réglé par le programme
        sys.setrecursionlimit(self.precision)

        # On définit la taille des indicateurs visuels / taille de l'image
        self.minsize = int(video.Framessize[1] / 200)
        self.crosswidth = int(video.Framessize[0] / 500)
        self.rectanglewidth = int(video.Framessize[1] / 1250)
        self.resFps = 60

class Video(object):

    def __init__(self):

        self.paths = paths

        self.id = None  # titre de la vidéo
        self.videoinput()

        # self.Framerate = self.get_framerate() # nombre de frame par seconde
        self.Framerate = 240
        self.Framessize = self.get_framessize()  # taille des frames
        self.Frames = self.get_frames()  # liste contenant les frames de la vidéo

        self.markerscolor = None  # couleur des repères visuels sur la video
        self.orientation = None  # orientation de la video (paysage ou portrait)
        self.lenref = None  # longueur de référence associée à la video

        self.scale = None  # rapport distance sur nombre de pixels
        self.markercount = 0  # nombre de repères détectés sur la vidéo
        self.markers = []
        self.computationDuration = None  # temps mis par l'algorythme pour effectuer le traitement

        self.settings = Settings(self)  # réglages associés à la vidéo
        self.treatementEvents = ''

    def videoinput(self) -> None:
        """
        Pas d'argument.

        Récupère la vidéo auprès de l'utilisateur.
        """
        self.paths.create_dir('bac')
        isempty = True
        print(mess.B_vi, end='')
        while isempty:
            if len(os.listdir(self.paths.bac)) != 0:
                isempty = False
            t.sleep(0.5)
        _bac = os.listdir(self.paths.bac)
        ext = _bac[0].split('.')[1]
        if len(_bac) == 1 and (ext == 'mp4' or ext == 'mov'):
            video = _bac[0]
            self.paths.videoinput = self.paths.bac + '/' + video
            self.paths.create_dir('videoStorage')
            sht.copy2(self.paths.videoinput, self.paths.videoStorage)
            self.id = str(video)
            self.paths.delete_dir('bac')
            return None
        elif len(_bac) == 1 and ext != 'mp4' and ext != 'mov':
            print(mess.P_vi1, end='')
            source = self.paths.bac + '/' + _bac[0]
            destination = self.paths.desktop + '/' + _bac[0]
            sht.copy2(source, destination)
            self.paths.delete_dir('bac')
            self.videoinput()
        elif len(_bac) > 1:
            print(mess.P_vi2, end='')
            self.paths.delete_dir('bac')
            self.videoinput()

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
        media_info = mi.MediaInfo.parse(self.paths.videoStorage + '/' + self.id)
        video_tracks = media_info.video_tracks[0]
        w = int(video_tracks.sampled_width)
        h = int(video_tracks.sampled_height)
        framessize = (w, h)
        return framessize

    def get_frames(self) -> list:
        """
        Pas d'argument.

            Renvoie une liste contenant l'ensemble des frames (tableaux de type
        uint8) dans le même ordre que dans la vidéo étudiée.
        """
        frames = []
        cam = cv2.VideoCapture(self.paths.videoStorage + '/' + self.id)
        frame_number = 0
        print(mess.B_gf, end='')
        while True:
            ret, frame = cam.read()
            if ret:
                frames.append(Frame(frame_number, np.array(frame)))
                frame_number += 1
            else:
                break
        cam.release()
        cv2.destroyAllWindows()
        print(mess.E_gf, end='')
        return frames


class Frame(object):
    def __init__(self, id, array):
        self.id = id
        self.array = array
        self.mesures = []
        self.identifiedObjects = []


class Object(object):
    def __init__(self, id, initpos, initframe):
        self.id = id
        self.lastupdate = 0
        self.lastknownpos = initpos
        self.predictions = {initframe: initpos}
        self.positions = {initframe: initpos}
        self.kf = KallmanFilter(initpos)
        self.status = 'hooked'


class Mesure(object):
    def __init__(self, pos):
        self.pos = pos
        self.status = 'unmatched'


class KallmanFilter(object):
    # filtre de kalman 
    def __init__(self, point):

        # Vecteur d'etat initial
        self.E = np.matrix([[point[0]], [point[1]], [0], [0]])

        # Matrice de transition
        self.A = np.matrix([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Matrice d'observation, on n'observe que x et y.
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.Q = np.matrix([[200, 0, 0, 0],
                            [0, 10, 0, 0],
                            [0, 0, 200, 0],
                            [0, 0, 0, 10]])

        self.R = np.matrix([[1, 0],
                            [0, 1]])

        self.P = np.eye(self.A.shape[1])

    def predict(self):
        self.E = np.dot(self.A, self.E)
        # Calcul de la covariance de l'erreur
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.E

    def update(self, z):
        # Calcul du gain de Kalman
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Correction / innovation
        self.E = np.round(self.E + np.dot(K, (z - np.dot(self.H, self.E))))
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P

        return self.E


class Calib:

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def reboot(video: Video, i=0) -> None:
        video.settings.precision = 1000
        sys.setrecursionlimit(1000)
        video.settings.step = 1
        video.Frames[i].identifiedObjects = []
        video.markers = []
        return None


# Traitement tools

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
            tleft = waiting_time(frame.id, len(frames), Ti)
            print(mess.S_vt + progr + ' % (' + tleft + ')', end='')
            T = t.time()

    d = time_formater(t.time() - Ti)
    video.computationDuration = d

    print(mess.E_vt, end='')
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

                element_in = False
                for obj in extremas:
                    HorizontalAlignement = obj[1] <= j <= obj[3]
                    VerticalAlignement = obj[0] <= i <= obj[2]
                    if VerticalAlignement and HorizontalAlignement:
                        element_in = True

                if not element_in:
                    depart = [i, j]
                    object = [depart]
                    init_extr = [depart[0], depart[1], depart[0], depart[1]]
                    at_border = False
                    Ti = t.time()
                    res = border_detection(image, depart, object, init_extr, mc, tol)
                    s += t.time() - Ti
                    extremas.append(res[0])
                    borders.append(res[1])

    return extremas, borders, s

def rate_rgb(pixel: list, c: int) -> float:
    """
    pixel : élement de l'image d'origine sous la forme [r, g, b].
        c = 0(rouge), 1(vert) ou 2(bleu).

        Calcul le poids relatif de la composante c du pixel parmis les
    composantes rgb qui le définissent.
    """
    assert c in [0, 1, 2]
    s = int(pixel[0]) + int(pixel[1]) + int(pixel[2])
    if 600 > s > 150:
        return int(pixel[c] + 1) / (s + 3) * 100
    else:
        return 0.

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
    if start not in obj:  # but: récupérer un encadrement de l'objet
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
    
        Renvoie la liste des voisins du pixel 'pixel' à étudier dans le cadre 
    de la recherche d'objet.
    """
    global at_border
    x, y = pixel[0], pixel[1]
    h = len(image)
    w = len(image[0])
    view = 2

    neighbours_coordinates = []
    for i in range(-view, view + 1):
        for j in range(-view, view + 1):
            if j != i:
                neighbours_coordinates.append([(x + i) % w, (y + j) % h])

    is_border = False
    outsiders = []
    for n in neighbours_coordinates:
        if rate_rgb(image[n[1]][n[0]], mc) < tol:
            is_border = True
            at_border = True
            outsiders.append(n)

    L_neighbours = []
    if not is_border and not at_border:
        L_neighbours.append([pixel[0] + 1, pixel[1]])
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
    extremas : dictionaire contenant les coordonnées extremales des repères
        détectés sur une frame.
    
        Détermine la position d'un objet à partir des extremas.
    Renvoie un dictionnaire où les clefs sont les noms des différents objets
    détectés sur la frame étudiée et les valeurs sont les coordonées du
    'centre' de l'objet.
    """
    position = []
    for obj in extremas:
        x = (obj[0] + obj[2]) / 2
        y = (obj[1] + obj[3]) / 2
        position.append([x, y])
    return position


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
                obj.positions[frame.id] = obj.predictions[frame.id]
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


def waiting_time(i: int, N: int, Ti: float) -> str:
    d = t.time() - Ti
    d = round((N - i) * (d / i), 1)
    return time_formater(d)

def time_formater(t: float) -> str:
    """
    Met en forme la durée entrée en argument pour la rendre lisible par l'utilisateur
    """
    minutes = str(int(t // 60))
    if int(minutes) < 10:
        minutes = '0' + minutes
    secondes = str(int(t % 60))
    if int(secondes) < 10:
        secondes = '0' + secondes
    return minutes + 'min ' + secondes + 'sec'


calib = Calib()
