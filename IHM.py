import csv
import inspect
import time as t
import shutil as sht

import cv2
import numpy as np

from Base import Break, mess
from MainConstructor import rate_rgb
from MainConstructor import Video, Frame


class Visu:
    # La classe visu regroupe les méthodes qui permttent de visualiser les résultats produits par l'algorythme.
    # Elle est notamment utile lors de la phase de calibration ou de la création de la vidéo de rendu
    @staticmethod
    def copy_im(image: np.array) -> np.array:
        """
        image : tableau numpy.

        Copie l'image passée en argument de manière à casser le lien entre les objets.
        """
        h = len(image)
        w = len(image[0])
        newIm = []
        for y in range(h):
            newLine = []
            for x in range(w):
                newLine.append(image[y][x])
            newIm.append(newLine)
        return np.uint8(newIm)

    @staticmethod
    def reduced(mc: int, tol: float, image: np.array) -> np.array:
        """
        mc : markerscolor, couleur des repères de l'image étudiée.
        tol : seuil de détection des couleurs.
        definition : taux de réduction de l'image.
        image : image étudiée.

        Crée un apercu de ce que percoit l'algorythme.
        """
        h = len(image)
        w = len(image[0])
        newIm = []
        for j in range(0, h):
            newLine = []
            for i in range(0, w):
                if rate_rgb(image[j][i], mc) > tol:
                    newLine.append(255)
                else:
                    newLine.append(0)
            newIm.append(newLine)
        return np.uint8(newIm)

    @staticmethod
    def detection(image: np.array, borders: list, copy=False) -> np.array:
        """
        image : image étudiée.
        borders : contours des repères detectés.
        copy : optionel, permet de defaire le lien entre l'image créée et l'image original.

        Crée un apercu de ce que l'algorythme détecte.
        """
        if copy:
            image = np.copy(image)
        h, w = image.shape[:2]
        for j in range(h):
            for i in range(w):
                if image[j][i] == 255:
                    image[j][i] = 100
        for obj in borders:
            for pixel in obj:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if 0 <= pixel[1] < h - j and 0 <= pixel[0] < w - i:
                            image[pixel[1] + j][pixel[0] + i] = 255
        return np.uint8(image)

    @staticmethod
    def rectangle_NB(image: np.array, extremas: list, rectanglewidth: int) -> np.array:
        """
        image : image étudiée.
        extremas : coordonées extremales des repères.
        rectanglewidth : largeur du contour tracé autour des repères detectés.

        Crée un apercu de ce que detecte l'algorythme.
        """
        h = len(image)
        w = len(image[0])
        marge = 4
        for obj in extremas:
            xmin, ymin = int(obj[0]) - marge, int(obj[1]) - marge
            xmax, ymax = int(obj[2]) + marge, int(obj[3]) + marge
            for i in range(xmin - rectanglewidth, xmax + rectanglewidth + 1):
                for n in range(rectanglewidth + 1):
                    if 0 <= i < w and 0 <= ymin - n < h and 0 <= ymin + n < h:
                        image[(ymin - n) % h][i % w], image[(ymax + n) % h][i % w] = 255, 255
            for j in range(ymin - rectanglewidth, ymax + rectanglewidth + 1):
                for n in range(rectanglewidth + 1):
                    if 0 <= xmin - n < w and 0 <= xmin + n < w and 0 <= j < h:
                        image[j % h][(xmin - n) % w], image[j % h][(xmax + n) % w] = 255, 255
        return np.uint8(image)

    @staticmethod
    def cross_color(image: np.array, pos: list, crosswidth: int, mc, copy=False) -> np.array:
        """
        image : np.array, imaghe sur laquelle on veut ajouter les croix. pos : positions où l'on souhaite tracer les
        croix sous forme [[x, y]] crosswidth : largeur des traits de la croix (quelques pixels) copy : optional,
        indique s'il est necéssaire de défaire le lien entre l'image d'origine et l'image traitée par la suite.

        Trace les croix aux positions passées en argument.
        """
        if copy:
            image = np.copy(image)

        h = len(image)
        w = len(image[0])
        for obj in pos:
            x = int(obj[0])
            y = int(obj[1])
            for i in range(x - crosswidth * 10, x + crosswidth * 10 + 1):
                for n in range(y - int(crosswidth / 2), y + int(crosswidth / 2) + 1):
                    if 0 <= i < w and 0 <= n < h:
                        image[n][i] = [0, 0, 0]
                        image[n][i][mc] = 255
            for j in range(y - crosswidth * 10, y + crosswidth * 10 + 1):
                for n in range(x - int(crosswidth / 2), x + int(crosswidth / 2) + 1):
                    if 0 <= n < w and 0 <= j < h:
                        image[j][n] = [0, 0, 0]
                        image[j][n][mc] = 255
        return np.uint8(image)

    @staticmethod
    def scale(image: np.array, scale: float, crosswidth: int, mc: int) -> np.array:
        """
        image : image étudiée.
        scale : échelle de la vidéo.
        crosswidth : largeur des traits des croix tracées sur l'image.
        mc : markerscolor, couleur des repères sur l'image étudiée.

        Crée un apercu de l'échelle utilisée pour le traitement de la vidéo.
        """
        h = len(image)
        w = len(image[0])
        color = [0, 0, 0]
        color[mc] = 255
        for i in range(int(1 / scale)):
            for j in range(crosswidth):
                image[(j + h - int(h / 20)) % h][(i + int(w / 10)) % w] = color
        location = (int(w / 10), h - int(h / 20 + h / 100))
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 1
        cv2.putText(image, '1cm', location, font, size, color)
        return np.uint8(image)

    @staticmethod
    def pas(image: np.array, pas: int) -> np.array:
        if pas >= 2:
            for j in range(int(len(image) / pas)):
                for i in range(int(len(image[j]) / pas)):
                    image[j * pas][i * pas] = [0, 0, 0]
        return np.uint8(image)

    def visusCalib(self, video: Video, frame: Frame, borders: list, extremas: list) -> None:
        rw = video.settings.rectanglewidth
        cw = video.settings.crosswidth
        mc = video.markerscolor
        scale = video.scale
        tol = video.settings.tol

        print(mess.B_vis, end='')
        visualisations = []

        color_im = np.copy(frame.array)
        visualisations.append(color_im)

        NB_im = self.reduced(mc, tol, color_im)
        visualisations.append(NB_im)

        treated_NB = self.detection(NB_im, borders, copy=True)
        treated_NB = self.rectangle_NB(treated_NB, extremas, rw)
        visualisations.append(treated_NB)

        pos = [obj.positions[frame.id] for obj in frame.identifiedObjects]
        treated_color = self.cross_color(frame.array, pos, cw, (mc+1) % 3, copy=True)
        treated_color = self.scale(treated_color, scale, cw, mc)
        visualisations.append(treated_color)

        print(mess.S_vis, end='')

        # On présente les résultats à l'utilisateur.
        for im in visualisations:
            cv2.imshow('calibration window', im)
            cv2.waitKey(0)
            cv2.destroyWindow('calibration window')
            cv2.waitKey(1)

        print(mess.E_vis, end='')
        return None


class Download:
    # La classe download gère les méthodes permettant de télécharger les différents résultats produits par l'algorythme.
    # Elle permet également detélécharger les réglages avec lesquels la vidéo a été traitée.

    def results(self, video: Video) -> None:
        """
        Gère l'appel aux différentes fonctions de téléchargement
        """
        self.video(video)
        self.treatedVideo(video)
        # self.frames(video)
        return None

    @staticmethod
    def reboot(video: Video) -> None:
        """
        Efface les résultats obtens précédements dans le cas ou la video a déjà été étudiée.
        """
        video.paths.add_subdata_dirs(video.id)
        video.paths.delete_dir('csv')
        video.paths.delete_dir('frames')
        video.paths.delete_dir('videodl')
        video.paths.add_subdata_dirs(video.id)
        return None

    @staticmethod
    def video(video: Video) -> None:
        """
        Télécharge la vidéo
        """
        video.paths.create_dir('videodl')
        source = video.paths.videoStorage + '/' + video.id
        destination = video.paths.videodl + '/vidéo' + '.mp4'
        sht.copy2(source, destination)
        sht.rmtree(video.paths.videoStorage)
        return None

    @staticmethod
    def treatedVideo(video: Video) -> None:
        """
        Télécharge la vidéo avec les croix tracées dessus
        """

        crosswidth = video.settings.crosswidth
        mc = video.markerscolor
        path = video.paths.videodl + '/vidéo traitée.mp4'
        ext = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.Framerate

        out = cv2.VideoWriter(path, ext, fps, video.Framessize)
        print(mess.B_vdl, end='')
        for frame in video.Frames:

            pos = [obj.positions[frame.id] for obj in video.markers]
            pred = [obj.predictions[frame.id] for obj in video.markers if frame.id in obj.predictions]

            # img = visu.pas(img, pas)
            img = visu.cross_color(frame.array, pred, crosswidth, (mc + 2) % 3)
            img = visu.cross_color(img, pos, crosswidth, (mc+1) % 3)

            out.write(img)

        out.release()
        print(mess.E_vdl, end='\n')
        return None

    def data(self, video: Video) -> None:
        """
        Télécharge les positions occupées par les différents repères au cours du temps sous forme de tableau csv
        """
        video.paths.create_dir('csv')
        nom_colonnes = ['frame', 'time']
        frames = video.Frames
        for obj in video.markers:
            nom_colonnes += ['X' + obj.id, 'Y' + obj.id]
        dos = open(video.paths.csv + '/positions objets.csv', 'w')
        array = csv.DictWriter(dos, fieldnames=nom_colonnes)
        array.writeheader()
        for frame in frames:
            time = round(frame.id / video.Framerate, 3)
            dico = {'frame': ' ' + str(frame.id), 'time': ' ' + str(time)}
            for obj in video.markers:
                dico['X' + obj.id] = ' ' + str(video.scale * obj.positions[frame.id][0])
                dico['Y' + obj.id] = ' ' + str(video.scale * obj.positions[frame.id][1])
            array.writerow(dico)
        dos.close()

        self.settings(video)
        self.events(video)

        print(mess.E_ddl)
        return None

    @staticmethod
    def settings(video: Video) -> None:
        """
        Télécharge les réglages avec lesquels a été fait le traitement
        """
        settings = video.settings
        doc = open(video.paths.csv + '/settings.csv', 'w')

        doc.write('------SETTINGS------\n')
        for attr in inspect.getmembers(settings):
            if attr[0][0] != '_' and not inspect.ismethod(attr[1]):
                line = attr[0] + ' ' * (19 - len(attr[0])) + ' : ' + str(attr[1]) + '\n'
                doc.write(line)

        doc.write('\n-------VIDEO--------\n')
        toAvoid = ['markers', 'paths', 'treatementEvents', 'Frames', 'settings']
        for attr in inspect.getmembers(video):
            if attr[0][0] != '_' and not inspect.ismethod(attr[1]):
                if not attr[0] in toAvoid:
                    if attr[0] != 'markerscolor' and attr[0] != 'orientation':
                        line = attr[0] + ' ' * (19 - len(attr[0])) + ' : ' + str(attr[1]) + '\n'
                    elif attr[0] == 'markerscolor':
                        line = attr[0] + ' ' * (19 - len(attr[0])) + ' : ' + ['blue', 'green', 'red'][attr[1]] + '\n'
                    elif attr[0] == 'orientation':
                        line = attr[0] + ' ' * (19 - len(attr[0])) + ' : ' + ['landscape', 'portrait'][
                            attr[1] - 1] + '\n'
                    doc.write(line)
        doc.close()
        return None

    @staticmethod
    def events(video: Video) -> None:
        """
        Télécharge un compte des potentielles difficultés qu'a pu rencontrer l'algorythme lors du traitement
        """
        doc = open(video.paths.csv + '/events.csv', 'w')
        doc.write(video.treatementEvents)
        doc.close()
        return None

    @staticmethod
    def frames(video: Video) -> None:
        """
        Télecharge l'ensemble des frames de l'image séparement
        """
        mc = video.markerscolor
        video.paths.create_dir('non treated frames')
        video.paths.create_dir('treated frames')
        print('\nSauvegarde des frames en cours ...', end='')
        for frame in video.Frames:
            name = video.paths.NonTreatedFrames + str(frame.id) + '.jpg'
            cv2.imwrite(name, frame.array)
            name = video.paths.TreatedFrames + str(frame.id) + '.jpg'
            crosswidth = video.settings.crosswidth
            im = visu.cross_color(frame.array, frame.identified_objects, crosswidth, mc)
            cv2.imwrite(name, im)
        print(mess.E_fdl)
        return None


class Interact:
    # La classe interact regroupe les méthodes qui vont permettre à l'algorythme d'intéragir avec l'utilisateur
    def __init__(self):
        self.stoplist = ['stop', 'quit', 'abandon', 'kill']

    def verif_settings(self, video: Video) -> None:
        """
        Éffectue les changements de réglges demandés par l'utilisateur'
        """
        settings = video.settings
        print(mess.S_vs1 + ['bleue', 'verte', 'rouge'][video.markerscolor], end='')
        print(mess.S_vs2 + ['landscape', 'portrait'][video.orientation - 1], end='')
        print(mess.S_vs3 + str(video.lenref) + ' cm', end='')
        print(mess.S_vs4 + str(100 - settings.tol), end='')
        isOk = False
        while not isOk:
            which_L = input(mess.I_vs).split(',')
            isOk = True
            for which in which_L:
                if which in self.stoplist:
                    raise Break

                try:
                    which = int(which)
                    if which in [0, 1, 2, 3, 4]:
                        if which == 0:
                            pass
                        elif which == 1:
                            print()
                            self.markerscolor_input(video)
                        elif which == 2:
                            print()
                            self.orientation_input(video)
                        elif which == 3:
                            print()
                            self.ref_input(video)
                        elif which == 4:
                            print()
                            self.tol_input(video)
                    else:
                        isOk = False
                        print(mess.P_vs)
                except ValueError:
                    isOk = False
                    print(mess.P_vs)

        return None

    def yn(self, question: str) -> bool:
        """
        question : question posée à l'utilisateur

        Pose une question fermée à l'utilisateur et renvoie un booléen en fonction de sa réponse.
        """
        assert type(question) == str
        while True:
            yn = input(question + ' [y]/n : ')
            if yn in ['y', '', 'n']:
                if yn == 'y' or yn == '':
                    return True
                elif yn == 'n':
                    return False
            elif yn in self.stoplist:
                raise Break
            else:
                print(mess.P_vs)

    def markerscolor_input(self, video: Video) -> None:
        """
        Récupère auprès de l'utilisateur la couleur des repères placés sur l'objet étudiée sur la vidéo et assigne cette
        valeur à l'attribut markerscolor de la vidéo.
        """
        while True:
            c = input(mess.I_mco)
            if c in ['1', '2', '3']:
                c = int(c) - 1
                video.markerscolor = c
                return None
            elif c in self.stoplist:
                raise Break
            else:
                print(mess.P_vs, end='')

    def orientation_input(self, video: Video) -> None:
        """
        Récupère l'orientation de la vidéo auprès de l'utilisateur
        """
        Framessize = video.Framessize
        while True:
            mode = input(mess.I_orn)
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
            elif mode in self.stoplist:
                raise Break
            else:
                print(mess.P_vs, end='')

    def ref_input(self, video: Video) -> None:
        """
        Récupère au près de l'utilisateur la distances séparant les deux premiers repères placés sur l'objet étudiée
        sur la vidéo et assigne cette valeur à l'attribut lenref de la vidéo.
        """
        while True:
            l = input(mess.I_ref)
            try:
                if l in self.stoplist:
                    raise Break
                else:
                    lenref = float(l)
                    video.lenref = lenref
                    return None
            except ValueError:
                print(mess.P_vs, end='')

    def tol_input(self, video: Video):
        """
        Permet à l'utilisateur de régler la tolérence (poids seuil de la couleur des repères dans chque pixel)
        """
        settings = video.settings
        while True:
            tol = input('Tolérance actuelle : ' + str(100 - settings.tol) + ', implémenter de : ')
            if tol in self.stoplist:
                raise Break
            else:
                try:
                    tol = round(float(tol), 3)
                    settings.tol -= tol
                    return None
                except ValueError:
                    print(mess.P_vs, end='')

    def waiting_time(self, i: int, N: int, Ti: float) -> str:
        """
        i : indice de la frame actuellement traitée
        N : nombre de frames qui constituent la vidéo
        Ti : instant

        Détermine le temps restant pour compléter la tâche
        """
        d = t.time() - Ti
        d = round((N - i) * (d / i), 1)
        return self.time_formater(d)

    @staticmethod
    def time_formater(t: float) -> str:
        """
        t : durée en secondes à mettre au format ..min ..sec

        Met en forme la durée entrée en argument pour la rendre lisible par l'utilisateur
        """
        minutes = str(int(t // 60))
        if int(minutes) < 10:
            minutes = '0' + minutes
        secondes = str(int(t % 60))
        if int(secondes) < 10:
            secondes = '0' + secondes
        return minutes + 'min ' + secondes + 'sec'


visu = Visu()
download = Download()
interact = Interact()
