import os
import time as t
import shutil as sht

import pymediainfo as mi
import numpy as np
import cv2

from Base import paths, mess
from SettingsConstructor import Settings
from KallmanFilterConstructor import KallmanFilter


# Élement central de la detection
def rate_rgb(pixel: list, c: int) -> float:
    """
    pixel : élement de l'image d'origine sous la forme [r, g, b].
    c : 0(rouge), 1(vert) ou 2(bleu).

    Calcul le poids relatif de la composante c du pixel parmis les composantes rgb qui le définissent.
    """
    assert c in [0, 1, 2]
    s = int(pixel[0]) + int(pixel[1]) + int(pixel[2])
    if 600 > s > 150:
        return int(pixel[c] + 1) / (s + 3) * 100
    else:
        return 0.


class Video(object):

    def __init__(self):

        self.paths = paths

        self.id = None  # titre de la vidéo
        self.videoinput()

        self.Framerate = self.get_framerate()  # nombre de frame par seconde
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
        Récupère la vidéo auprès de l'utilisateur.
        """
        self.paths.create_dir('bac')
        acceptedFormats = ['mp4', 'mov', 'Mov']
        isempty = True
        print(mess.B_vi0, end='')
        while isempty:
            if len(os.listdir(self.paths.bac)) != 0:
                isempty = False
            t.sleep(0.5)
        _bac = os.listdir(self.paths.bac)
        ext = _bac[0].split('.')[1]
        if len(_bac) == 1 and ext in acceptedFormats:
            video = _bac[0]
            self.paths.videoinput = self.paths.bac + '/' + video
            self.paths.create_dir('videoStorage')
            sht.copy2(self.paths.videoinput, self.paths.videoStorage)
            self.id = str(video)
            self.paths.delete_dir('bac')
            return None
        elif len(_bac) == 1 and ext not in acceptedFormats:
            print(mess.P_vi1, end='')
            source = self.paths.bac + '/' + _bac[0]
            destination = self.paths.desktop + '/' + _bac[0]
            sht.copy2(source, destination)
            self.paths.delete_dir('bac')
            t.sleep(2)
            self.videoinput()
        elif len(_bac) > 1:
            print(mess.P_vi2, end='')
            self.paths.delete_dir('bac')
            t.sleep(2)
            self.videoinput()

    def get_framerate(self) -> float:
        """
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
        Renvoie un tuple de deux valeurs : la hauteur et largeur des frames de la vidéo.
        """
        media_info = mi.MediaInfo.parse(self.paths.videoStorage + '/' + self.id)
        video_tracks = media_info.video_tracks[0]
        w = int(video_tracks.sampled_width)
        h = int(video_tracks.sampled_height)
        framessize = (w, h)
        return framessize

    def get_frames(self) -> list:
        """
        Renvoie une liste contenant l'ensemble des frames (tableaux de type uint8) dans le même ordre que dans la vidéo
        étudiée.
        """
        frames = []
        cam = cv2.VideoCapture(self.paths.videoStorage + '/' + self.id)
        frame_number = 0
        print(mess.B_gfs, end='')
        while True:
            ret, frame = cam.read()
            if ret:
                frames.append(Frame(frame_number, np.array(frame)))
                frame_number += 1
            else:
                break
        cam.release()
        cv2.destroyAllWindows()
        print(mess.E_gfs, end='')
        return frames


class Frame(object):
    def __init__(self, id, array):
        self.id = id
        self.array = array
        self.mesures = []
        self.identifiedObjects = []


class Object(object):
    def __init__(self, id, initpos, initframe, dt):
        self.id = id
        self.lastupdate = 0
        self.lastknownpos = initpos
        self.predictions = {initframe: initpos}
        self.positions = {initframe: initpos}
        self.kf = KallmanFilter(dt, initpos)
        self.status = 'hooked'


class Mesure(object):
    def __init__(self, pos):
        self.pos = pos
        self.status = 'unmatched'
