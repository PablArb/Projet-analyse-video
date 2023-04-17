import sys


class Settings(object):
    def __init__(self, video):
        self.precision = 1000  # permet de gérer la precision du système
        self.maxPrec = 1e6  # plafond imposé à la limitte de récursion
        self.tol = 40.0  # est réglable lors de l'execution
        self.step = 1  # est automatiquement réglé par le programme

        # On définit la taille des indicateurs visuels / taille de l'image
        self.minsize = int(video.Framessize[1] / 200)
        self.crosswidth = int(video.Framessize[0] / 500)
        self.rectanglewidth = int(video.Framessize[1] / 1250)

        sys.setrecursionlimit(self.precision)