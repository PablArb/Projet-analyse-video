#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:53:12 2022

@author: pabloarb
"""
import os
import shutil as sht
import getpass as gp


class Break (Exception):
    pass

class SettingError (Exception):
    pass

# définition des paths utiles 
class Paths :
    def __init__(self):
        self.pathList = ['desktop','bac', 'videoStorage', 'data']
    
    def create_dir(self, dir:str) -> None :
        '''
        dir : nom du dossier à créer.
        Permet de créer le dossier dont le nom est passé en argument.
        '''
        attr = self.__dict__
        if dir in attr :
            p = attr[dir]
        else : 
            raise AttributeError
        if not os.path.exists(p):
            os.makedirs(p)
        return None
    
    def delete_dir(self, dir:str) -> None :
        '''
        dir : nom du dossier à supprimer.
        Permet de supprimer le dossier dont le nom est passé en argument.
        '''
        attr = self.__dict__
        if dir in attr :
           if os.path.exists(attr[dir]) :
               sht.rmtree(attr[dir])
        else : 
           raise AttributeError
        
        return None
    
    def add_subdata_dirs(self, video:str) -> None:
        '''
        video : nom de la video passée en entrée du script.
        Permet d'ajouter les dossier propre à la vidéo dans le dossier data
        (où les résultats de l'étude sont stockés).
        '''
        self.csv = self.data + '/' + video + '/csv'
        self.videodl = self.data + '/' + video + '/vidéo'
        self.frames = self.data + '/' + video + '/frames'
        self.TreatedFrames = self.frames + '/treated'
        self.NonTreatedFrames = self.frames + '/non treated'
        return None

class MacosPaths (Paths):
    def __init__(self):
        self.desktop = '/Users/'+user+'/Desktop'
        self.bac = '/Users/'+user+'/Desktop/bac'
        self.videoStorage = '/Users/'+user+'/.##temporary storage##'
        self.data = '/Users/'+user+'/Desktop/mes exp TIPE/data video'
   
class WIndowsPaths (Paths):
    def __init__(self):
        self.desktop = 'C:Users/'+user+'/Desktop'
        self.bac = 'C:/Users/'+user+'/Desktop/bac'
        self.videoStorage = 'C:/Users/'+user+'/Desktop/.##temporary storage##'
        self.data = '/C:Users/'+user+'/Desktop/TIPE/data video'

user = gp.getuser()
if os.name == 'nt':
    paths = WIndowsPaths()
elif os.name == 'posix':
    paths = MacosPaths()
else :
    pass