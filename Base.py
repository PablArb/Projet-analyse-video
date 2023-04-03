#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:53:12 2022

@author: pabloarb
"""
import os
import shutil as sht
import getpass as gp


# définition des paths utiles
class Paths:
    def __init__(self):
        self.TreatedFrames = None
        self.frames = None
        self.NonTreatedFrames = None
        self.videodl = None
        self.csv = None
        self.pathList = ['desktop', 'bac', 'videoStorage', 'data']

    def create_dir(self, dir: str) -> None:
        """
        dir : nom du dossier à créer.

        Crée le dossier dont le nom est passé en argument.
        """
        attr = self.__dict__
        if dir in attr:
            p = attr[dir]
        else:
            raise AttributeError
        if not os.path.exists(p):
            os.makedirs(p)
        return None

    def delete_dir(self, dir: str) -> None:
        """
        dir : nom du dossier à supprimer.

        Supprime le dossier dont le nom est passé en argument.
        """
        attr = self.__dict__
        if dir in attr:
            if os.path.exists(attr[dir]):
                sht.rmtree(attr[dir])
        else:
            raise AttributeError

        return None

    def add_subdata_dirs(self, video: str) -> None:
        """
        video : nom de la video passée en entrée du script.

            Ajoute les dossiers propre à la vidéo dans le dossier data (où les
        résultats de l'étude sont stockés).
        """
        self.csv = self.data + '/' + video + '/csv'
        self.videodl = self.data + '/' + video + '/vidéo'
        self.frames = self.data + '/' + video + '/frames'
        self.TreatedFrames = self.frames + '/treated'
        self.NonTreatedFrames = self.frames + '/non treated'
        return None


class MacosPaths(Paths):
    def __init__(self):
        super().__init__()
        self.desktop = '/Users/'+user+'/Desktop'
        self.bac = '/Users/'+user+'/Desktop/bac'
        self.videoStorage = '/Users/'+user+'/.##temporary storage##'
        self.data = '/Users/'+user+'/Desktop/mes exp TIPE/A - data video'


class WIndowsPaths (Paths):
    def __init__(self):
        super().__init__()
        self.desktop = 'C:Users/' + user + '/Desktop'
        self.bac = 'C:/Users/' + user + '/Desktop/bac'
        self.videoStorage = 'C:/Users/' + user + '/Desktop/.##temporary storage##'
        self.data = '/C:Users/' + user + '/Desktop/TIPE/data video'


class Mess:
    def __init__(self):
        # B:begining, E:End, P:problem, I:input, S:info

        self.B_proc = '\rInitialisation de la procédure\n\n'
        self.E_proc = '\n\nProcédure terminée'

        self.B_vi = '\rPlacez la vidéo à étudier dans le bac sur votre bureau.'
        self.P_vi1 = '\rVeuillez fournir une vidéo au format mp4 ou mov'
        self.P_vi2 = "\rVeuillez ne placer qu'un document dans le bac"
        self.B_gf = '\rRécupération de la vidéo en cours ...'
        self.E_gf = '\rRécupération de la vidéo ------------------------------------------ OK\n\n'

        self.I_mc = 'Couleur des repères à étudier (1=bleu, 2=vert, 3=rouge) : '
        self.I_or = 'La vidéo est en mode (1=landscape, 2=portrait) : '
        self.I_ref = 'Longueur entre les deux premiers repères(cm) : '

        self.B_cal = '\nTraitement en cours ...'
        # potential recursion limit issue
        # potential setting issue
        self.E_cal = '\rTraitement -------------------------------------------------------- OK\n'

        self.B_vis = 'Création des visuels en cours ...'
        self.S_vis = "\rAffichage du résultat (une fenêtre a dû s'ouvrir)"
        self.E_vis = '\rValidation du résultat -------------------------------------------- OK\n'
        self.S_dur = 'Estimation de la durée du traitement : '

        self.I_val = '\n\nLe traitement est-il bon ?'
        # potential setting issue
        self.P_set = '\rproblèmes dans les réglages'

        self.S_vt = '\rTraitement en cours : '
        self.E_vt = '\rTraitement de la vidéo -------------------------------------------- OK\n'
        self.S_dvt = 'Durée du traitement : '

        self.I_dl = "Voulez vous télécharger les résultats de l'étude ?"
        self.E_ddl = '\n\nSauvegarde de la data --------------------------------------------- OK'
        self.B_vdl = 'Sauvegarde de la vidéo en cours ...'
        self.E_vdl = '\rSauvegarde de la vidéo -------------------------------------------- OK'
        self.E_fdl = '\nSauvegarde des frames --------------------------------------------- OK'

        # dealing with recursion limit
        self.P_rec = '\rDéfinition trop élevée, tentative avec une définition plus faible'

        # dealing with setting issue
        self.P_set = '\rIl y a un problème, veuillez vérifiez les réglages'
        self.S_vs1 = '\n1 couleur des repères : '
        self.S_vs2 = '\n2 orientation de la vidéo : '
        self.S_vs3 = '\n3 longueur de référence : '
        self.S_vs4 = '\n4 tolérance : '
        self.I_vs = '\nréglage qui vous semble éroné (0=aucun, 1, 2, 3, 4) : '
        self.P_vs = 'vous devez avoir fait une erreur, veuillez réessayer\n'
mess = Mess()

class Break(Exception):
    pass


class SettingError(Exception):
    print(mess.P_set, end='')
    pass


user = gp.getuser()
if os.name == 'nt':
    paths = WIndowsPaths()
elif os.name == 'posix':
    paths = MacosPaths()
else:
    pass
