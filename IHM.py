#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:06:13 2022

@author: pabloarb
"""

import csv, cv2, inspect, sys
import numpy as np
import shutil as sht
from Base import Break, mess
from VideoTreatment import Video 


class Visu :
    
    def copy_im(self, image:np.array) -> np.array:
        '''
        image : tableau numpy.
        
            Copie l'image passée en argument de manière a casser le lien entre
        les objets.
        '''
        h = len(image)
        w = len(image[0])
        newIm = []
        for y in range (h):
            newLine = []
            for x in range(w):
                newLine.append(image[y][x])
            newIm.append(newLine)
        return np.uint8(newIm)
    
    def cross_color(self, image:np.array, pos:list, crosswidth:int, copy=False) -> np.array:
        '''
        image       : np.array, imaghe sur laquelle on veut ajouter les croix.
        pos         : positions ou l'on veut tracer ces croix sous forme [[x, y]]
        crosswidth  : largeur des traits e la croix (qq pixels)
        copy        : optional, indique s'il est necéssaire de défaire le lien
            entre l'image d'origine et l'image traitée par la suite.
        
        Trace les croix aux positions des repères detectés.
        '''
        if copy:
            image = self.copy_im(image)

        h = len(image)
        w = len(image[0])
        for obj in pos :
            x = int(obj[0])
            y = int(obj[1])
            for i in range(x - crosswidth * 10, x + crosswidth * 10 + 1):
                for n in range(y - int(crosswidth / 2), y + int(crosswidth / 2) + 1):
                    if 0<=i<w and 0<=n<h :
                        image[n][i] = [0, 255, 0]
            for j in range(y - crosswidth * 10, y + crosswidth * 10 + 1):
                for n in range(x - int(crosswidth / 2), x + int(crosswidth / 2) + 1):
                    if 0 <= n < w and 0 <= j < h :
                        image[j][n] = [0, 255, 0]
        return np.uint8(image)
    
    def rate_rgb(self, pixel:list, c:int) -> float:
        assert c in [0, 1, 2]
        return int(pixel[c]) / (int(pixel[0]) + int(pixel[1]) + int(pixel[2]) + 1) * 100
    
    def reduced(self, mc:int, tol:float, definition:int, image:np.array) -> np.array:
        '''
        mc          : markerscolor, couleur des repères de l'image étudiée.
        tol         : seuil de détection des couleurs.
        definition  : taux de réduction de l'image.
        image       : image étudiée.

        Crée un apercu de ce que percoit l'algorythme.
        '''
        h = len(image)
        w = len(image[0])
        newIm = []
        for j in range(0, h, definition):
            newLine = []
            for i in range(0, w, definition):
                if self.rate_rgb(image[j][i], mc) > tol:
                    newLine.append(255)
                else :
                    newLine.append(0)
            newIm.append(newLine)
        return np.uint8(newIm)
    
    def rectangle_NB(self, image:np.array, extremas:dict, rectanglewidth:int) -> np.array:
        '''
        image           : image étudiée.
        extremas        : coordonées extremales des repères.
        rectanglewidth  : largeur du contour tracé autour des repères detectés.

        Crée un apercu de ce que detecte l'algorythme.
        '''
        h = len(image)
        w = len(image[0])
        marge = 4
        for obj in extremas:
            xmin, ymin = int(obj[0])-marge, int(obj[1])-marge
            xmax, ymax = int(obj[2])+marge, int(obj[3])+marge
            for i in range(xmin - rectanglewidth, xmax + rectanglewidth + 1):
                for n in range(rectanglewidth + 1):
                    if 0 <= i < w and 0 <= ymin-n < h and 0 <= ymin+n < h :
                        image[(ymin - n) % h][i % w], image[(ymax + n) % h][i % w] = 255, 255
            for j in range(ymin - rectanglewidth, ymax + rectanglewidth + 1):
                for n in range(rectanglewidth + 1):
                    if 0 <= xmin-n < w and 0 <= xmin+n < w and 0 <= j < h :
                        image[j % h][(xmin - n) % w], image[j % h][(xmax + n) % w] = 255, 255
        return np.uint8(image)
    
    def pas (self, image:np.array, pas:int) -> np.array:
        if pas >= 2 :
            for j in range (int(len(image)/pas)):
                for i in range (int(len(image[j])/pas)):
                    image[j*pas][i*pas] = [0, 0, 0]
        return np.uint8(image)
    
    def scale(self, image:np.array, scale:float, crosswidth:int, mc:int) -> np.array:
        '''
        image       : image étudiée.
        scale       : échelle de la vidéo.
        crosswidth  : largeur des traits des croix tracées sur la l'image.
        mc          : markerscolor, couleur des repères sur l'image étudiée.
                
        Crée un apercu de l'échelle utilisée pour le traitement de la vidéo.
        '''
        h = len(image)
        w = len(image[0])
        color = [0, 0, 0]
        color[mc] = 255
        for i in range(int(1/scale)):
            for j in range(crosswidth):
                image[j+h-int( h/20 )][i + int( w/10 )] = color
        location = (int(w/10) , h-int(h/20 + h/100))
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 1
        cv2.putText(image, '1cm', location, font, size, color)
        return np.uint8(image)
    
    def detection (self, image:np.array, borders:list, copy=False) -> np.array:
        '''
        image    : image étudiée.
        borders  : contours des repères detectés.
        copy     : optionel, permet de defaire le lien entre l'image créée et 
            l'image original.
            
        Crée un apercu de ce que l'algorythme détecte.
        '''
        if copy :
            image = self.copy_im(image)
        h , w = image.shape[:2]
        for j in range(h) :
            for i in range(w):
                if image[j][i] == 255:
                    image[j][i] = 100
        for obj in borders:
            for pixel in obj :
                for i in range (-1, 2):
                    for j in range (-1, 2):
                        if 0 <= pixel[1] < h-j and 0 <= pixel[0] < w-i :
                            image[pixel[1]+j][pixel[0]+i] = 255
        return np.uint8(image)



class Download :
    
    def results(self, video:Video) -> None :
        self.video(video)
        self.treatedVideo(video)
        # framesdownload(video)
        return None
    
    def reboot(self, video:Video) -> None:
        video.paths.add_subdata_dirs(video.id)
        video.paths.delete_dir('csv')
        video.paths.delete_dir('frames')
        video.paths.delete_dir('videodl')
        video.paths.add_subdata_dirs(video.id)
        return None
    
    def video(self, video:Video) -> None:
        video.paths.create_dir('videodl')
        source = video.paths.videoStorage  + '/' + video.id
        destination = video.paths.videodl + '/vidéo' + '.mp4'
        sht.copy2(source, destination)
        sht.rmtree(video.paths.videoStorage)
        return None
    
    def treatedVideo(self, video:Video) -> None:
        crosswidth = video.settings.crosswidth
        path = video.paths.videodl + '/vidéo traitée.mp4'
        ext = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, ext, video.Framerate, video.Framessize)
        print(mess.B_vdl, end='')
        for frame in video.Frames:
            pos = [obj.positions[frame.id] for obj in frame.identifiedObjects]
            img = visu.cross_color(frame.array, pos, crosswidth)
            # img = Add_pas(img, pas)
            out.write(img)
        out.release()
        print(mess.E_vdl, end='\n')
        return None
    
    def data(self, video:Video) -> None:
        video.paths.create_dir('csv')
        nom_colonnes = ['frame', 'time']
        frames = video.Frames
        for obj in video.markers :
            nom_colonnes += ['X' + obj.id, 'Y' + obj.id]
        dos = open(video.paths.csv + '/positions objets.csv', 'w')
        array = csv.DictWriter(dos, fieldnames=nom_colonnes)
        array.writeheader()
        for frame in frames:
            time = round(frame.id / video.Framerate, 3)
            dico = {'frame': ' ' + str(frame.id), 'time': ' ' + str(time)}
            for obj in video.markers :
                dico['X' + obj.id] = ' ' + str(video.scale * obj.positions[frame.id][0])
                dico['Y' + obj.id] = ' ' + str(video.scale * obj.positions[frame.id][1])
            array.writerow(dico)
        dos.close()
    
        self.settings(video)
        self.events(video)
        
        print(mess.E_ddl)
        return None
    
    def settings(self, video:Video) -> None:
        
        settings = video.settings
        doc = open(video.paths.csv + '/settings.csv', 'w')
    
        doc.write('------SETTINGS------\n')
        for attr in inspect.getmembers(settings):
            if attr[0][0] != '_' and not inspect.ismethod(attr[1]):
                line = attr[0] + ' '*(19-len(attr[0])) + ' : ' + str(attr[1]) + '\n'
                doc.write(line)
    
        doc.write('\n-------VIDEO--------\n')
        toAvoid = ['markers', 'paths', 'treatementEvents', 'Frames', 'settings']
        for attr in inspect.getmembers(video):
            if attr[0][0] != '_' and not inspect.ismethod(attr[1]):
                if not attr[0] in toAvoid :
                    if attr[0] != 'markerscolor':
                        line = attr[0] + ' '*(19-len(attr[0])) + ' : ' + str(attr[1]) + '\n'
                    else :
                        line = attr[0] + ' '*(19-len(attr[0])) + ' : ' + ['blue', 'green', 'red'][attr[1]] + '\n'
                    doc.write(line)
        doc.close()
        return None
    
    def events(self, video:Video):
        doc = open(video.paths.csv + '/events.csv', 'w')
        doc.write(video.treatementEvents)
        doc.close()
        return None
    
    def frames(self, video:Video) -> None:
        video.paths.create_dir('non treated frames')
        video.paths.create_dir('treated frames')
        print('\nSauvegarde des frames en cours ...', end='')
        for frame in video.frames:
            name = video.paths.NonTreatedFrames + str(frame.id) + '.jpg'
            cv2.imwrite(name, frame.array)
            name = video.paths.TreatedFrames + str(frame.id) + '.jpg'
            crosswidth = video.settings.crosswidth
            im = visu.cross_color(frame.array, frame.identified_objects, crosswidth)
            cv2.imwrite(name, im)
        print(mess.E_fdl)
        return None

class Interact :
    
    def __init__(self):
        self.stoplist = ['stop', 'quit', 'abandon', 'kill']
    
    def verif_settings(self, video:Video):
        settings = video.settings
        while True :
            print(mess.S_vs1 + ['bleue', 'verte', 'rouge'][video.markerscolor], end='')
            print(mess.S_vs2 + ['landscape', 'portrait'][video.orientation-1], end='')
            print(mess.S_vs3 + str(video.lenref) + ' cm', end='')
            print(mess.S_vs4 + str(100-settings.tol), end='')
            which = input(mess.I_vs)
            if which in ['0', '1', '2', '3', '4', 'pres']:
                if which == '0':
                    pass
                elif which == '1':
                    print()
                    self.markerscolor_input(video)
                elif which == '2':
                    print()
                    self.orientation_input(video)
                elif which == '3':
                    print()
                    self.ref_input(video)
                elif which == '4':
                    print()
                    self.tol_input(video)
                elif which == 'pres':
                    print()
                    self.reclimit_input(video)
                return None
            elif which in self.stoplist :
                raise Break
            else:
                print (mess.P_vs)
                
    def yn(self, question:str) -> bool :
        '''
        question : question posée à l'utilisateur
        
            Pose une question fermée à l'utilisateur et renvoie un booléen en 
        fonction de sa réponse.
        '''
        assert type(question) == str
        while True:
            yn = input(question + ' [y]/n : ')
            if yn in ['y', '', 'n']:
                if yn == 'y' or yn == '':
                    return True
                elif yn == 'n':
                    return False
            elif yn in self.stoplist :
                raise Break
            else:
                print(mess.P_vs)
    
    def markerscolor_input(self, video:Video) -> None:
        """
        Récupère au près de l'utilisateur la couleur des repères placés sur
        l'objet étudiée sur la vidéo et assigne cette valeur à l'attribut
        markerscolor de la vidéo.
        """
        while True :
            c = input(mess.I_mc)
            if c in ['1', '2', '3']:
                c = int(c)-1
                video.markerscolor = c
                return None
            elif c in self.stoplist :
                raise Break
            else:
                print(mess.P_vs, end='')

    def orientation_input(self, video:Video) -> None:
            Framessize = video.Framessize
            while True:
                mode = input(mess.I_or)
                if mode in ['1', '2']:
                    if mode == '1':
                        height = min(Framessize)
                        width = max(Framessize)
                    elif mode == '2':
                        height = max(Framessize)
                        width = min(Framessize)
                    Framessize = (width, height)
                    video.Framessize = Framessize
                    video.orientation = int(mode)
                    return None
                    break
                elif mode in self.stoplist :
                    raise Break
                else:
                    print(mess.P_vs, end='')

    def ref_input(self, video:Video) -> None:
        """
        Récupère au près de l'utilisateur la distances séparant les deux
        premiers repères placés sur l'objet étudiée sur la vidéo et assigne
        cette valeur à l'attribut lenref de la vidéo.
        """
        while True:
            l = input(mess.I_ref)
            try :
                if l in interact.stoplist:
                    raise Break
                else :
                    lenref = float(l)
                    video.lenref = lenref
                    return None
            except ValueError :
                print(mess.P_vs, end='')
                
    def tol_input(self, video:Video):
        settings = video.settings
        while True :
            tol = input('Tolérance actuelle : ' + str(100-settings.tol) + ', implémenter de : ')
            if tol in self.stoplist :
                raise Break
            else :
                try :
                    tol = round(float(tol), 3)
                    settings.tol -= tol
                    return None
                except ValueError :
                    print(mess.P_vs, end='')
        
    def reclimit_input(self, video:Video):
        while True :
            rl = input('setrecursionlimit : ')
            if rl in self.stoplist :
                raise Break
            else :
                try :
                    rl = int(rl)
                    sys.setrecursionlimit(rl)
                    return None
                except ValueError :
                    print(mess.P_vs, end='')
    
    
visu = Visu()
download = Download()
interact = Interact()