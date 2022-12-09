#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 16:32:07 2022

@author: pabloarb
"""

import sys, os, cv2
import time as t
import shutil as sht
import pymediainfo as mi
from ERRORS import paths



class Settings:
    def __init__(self, video):

        self.precision = 100       # permet de gérer la precision du système
        self.tol = 0.4             # est réglable lors de l'execution
        self.maxdef = 15           # abaissement de la definition maximal
        self.definition = 1        # est automatiquement réglé par le programme
        self.step = 1              # est automatiquement réglé par le programme
        sys.setrecursionlimit(self.precision)

        # On définit la taille des indicateurs visuels / taille de l'image
        self.minsize = int(video.Framessize[1] / 170)
        # self.maxdist = int(video.Framessize[1] / (0.25 * video.Framerate) * 5)
        # self.bordure_size = int(video.Framessize[0] /  video.Framerate * 2)
        self.maxdist = int(video.Framessize[1] / video.Framerate * 10)
        self.bordure_size = 0
        self.crosswidth = int(video.Framessize[1] / 500)
        self.rectanglewidth = int(video.Framessize[1] / 1250)




class Video(object):

    def __init__(self):

        self.paths = paths
        
        self.id = None
        self.name = None # id où l'on a supprimé le 'é' souvent présent dans vidéo
        self.videoinput()

        self.Frames = self.get_frames()
        self.Framerate = self.get_framerate()
        self.Framessize = self.get_framessize()

        self.markerscolor = None
        self.orientation = None
        self.lenref = None

        self.scale = None
        self.markercount = None
        self.computationDuration = None

        
        
    def videoinput(self) -> None:
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
            self.name = ''.join( tuple( video.split('́') ) )
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
        Renvoie une listes contenatnt l'ensembles des frames (tableaux de type
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
        Renvoie le nombre de frames par secondes de la vidéo passée en entrée du
        script.
        """
        media_info = mi.MediaInfo.parse(self.paths.videoStorage + '/' + self.id)
        tracks = media_info.tracks
        for i in tracks:
            if i.track_type == 'Video':
                framerate = float(i.frame_rate)
        return framerate

    def get_framessize(self) -> tuple:
        """
        Renvoie un tuple de deux valeurs : la hauteur et largeur des frames de
        la video.
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
