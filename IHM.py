#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:06:13 2022

@author: pabloarb
"""

import csv, os, cv2, inspect
import shutil as sht
import getpass as gp
import time as t


class Video(object):
    pass
class Settings(object):
    pass
class Break(Exception):
    pass



stoplist = ['stop', 'quit', 'abandon', 'kill']
user = gp.getuser()

def yn(question:str) -> bool :
    assert type(question) == str
    while True:
        yn = input('\n' + question + ' [y]/n : ')
        if yn in ['y', '', 'n']:
            if yn == 'y' or yn == '':
                return True
            elif yn == 'n':
                return False
        elif yn in stoplist :
            raise Break
        else:
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')




# définition des paths utiles 
class Paths :
    
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
           p = attr[dir]
        else : 
           raise AttributeError
        if os.path.exists(p) :
            sht.rmtree(p)
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
        self.pathList = ['bac', 'videoStorage', 'data']
        self.bac = '/Users/' + user + '/Desktop/bac'
        self.videoStorage = '/Users/' + user + '/Desktop/.##temporary storage##'
        self.data = '/Users/' + user + '/Desktop/TIPE/data video'
   
class WIndowsPaths (Paths):
    def __init__(self):
        self.pathList = ['bac', 'videoStorage', 'data']
        self.bac = 'C:/Users/' + user + '/Desktop/bac'
        self.videoStorage = 'C:/Users/' + user + '/Desktop/.##temporary storage##'
        self.data = '/C:Users/' + user + '/Desktop/TIPE/data video'


if os.name == 'nt':
    paths = WIndowsPaths()
elif os.name == 'posix':
    paths = MacosPaths()
else :
    pass




# Créations des visuels





# Récupération des résultats du traitement
def resultsdownload(video:Video, crosswidth:int) -> None :
    videodownload(video)
    create_video(video, crosswidth)
    # framesdownload(video, crosswidth)
    return None

def reboot(video:Video) -> None:
    video.paths.add_subdata_dirs(video.id)
    video.paths.delete_dir('csv')
    video.paths.delete_dir('frames')
    video.paths.delete_dir('vidéodl')
    video.paths.add_subdata_dirs(video.id)
    return None

def videodownload(video:Video) -> None:
    video.paths.create_dir('vidéodl')
    source = video.paths.videoStorage  + '/' + video.id
    destination = video.paths.vidéodl + '/vidéo' + '.mp4'
    sht.copy2(source, destination)
    sht.rmtree(video.paths.videoStorage)
    return None

def datadownload(video:Video, settings:Settings) -> None:
    video.paths.create_dir('csv')
    print('Sauvegarde de la data en cours ...', end='')
    nom_colonnes = ['frame', 'time']
    objects = []
    frames = video.Frames
    for frame in frames:
        for obj in frame.identifiedObjects:
            if obj not in objects:
                objects.append(obj)
                nom_colonnes += ['X' + obj.id, 'Y' + obj.id]
    dos = open(video.paths.csv + '/positions objets.csv', 'w')
    array = csv.DictWriter(dos, fieldnames=nom_colonnes)
    array.writeheader()
    for frame in frames:
        time = round(int(frame.id.split('.')[1]) / video.Framerate, 3)
        dico = {'frame': ' ' + frame.id, 'time': ' ' + str(time)}
        for obj in frame.identifiedObjects:
            dico['X' + obj.id] = ' ' + str(video.scale * obj.positions[frame.id][0])
            dico['Y' + obj.id] = ' ' + str(video.scale * obj.positions[frame.id][1])
        array.writerow(dico)
    dos.close()
    t.sleep(1)

    settingsdownload(video, settings)

    print('\rSauvegarde de la data --------------------------------------------- OK')
    return None

def settingsdownload(video:Video, settings:Settings) -> None:

    doc = open(video.paths.csv + '/settings.csv', 'w')

    doc.write('------SETTINGS------\n')
    for attr in inspect.getmembers(settings):
        if attr[0][0] != '_' and not inspect.ismethod(attr[1]):
            line = attr[0] + ' '*(19-len(attr[0])) + ' : ' + str(attr[1]) + '\n'
            doc.write(line)

    doc.write('\n-------VIDEO--------\n')
    for attr in inspect.getmembers(video):
        if attr[0][0] != '_' and not inspect.ismethod(attr[1]):
            if not attr[0] == 'Frames':
                line = attr[0] + ' '*(19-len(attr[0])) + ' : ' + str(attr[1]) + '\n'
                doc.write(line)
    doc.close()
    return None

def framesdownload(video:Video, crosswidth:int) -> None:
    video.paths.create_dir('non treated frames')
    video.paths.create_dir('treated frames')
    print('\nSauvegarde des frames en cours ...', end='')
    for frame in video.frames:
        name = video.paths.NonTreatedFrames + '/frame' + str(int(frame.id.split('.')[1])) + '.jpg'
        cv2.imwrite(name, frame.array)
        name = video.paths.TreatedFrames + '/frame' + str(int(frame.id.split('.')[1])) + '.jpg'
        im = draw_cross_color(frame.array, frame.identified_objects, crosswidth)
        cv2.imwrite(name, im)
    print('\rSauvegarde des frames --------------------------------------------- OK')
    return None

def create_video(video:Video, crosswidth:int) -> None:
    out = cv2.VideoWriter(video.paths.videodl + '/vidéo traitée' + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video.Framerate, video.Framessize)
    print()
    print('Sauvegarde de la vidéo en cours ...', end='')
    for frame in video.Frames:
        img = draw_cross_color(frame.array, frame, crosswidth)
        # img = Add_pas(img, pas)
        out.write(img)
    out.release()
    print('\rSauvegarde de la vidéo -------------------------------------------- OK', end='\n')
    return None

