import sys

class Settings(object):
    def __init__(self, video):
        self.modifiables = ('maxPrec', 'tol', 'maxBrightness', 'minBrightness', 'marge', 'view')

        self.precision = 1000  # permet de gérer la precision du système
        self.maxPrec = 1e6  # plafond imposé à la limitte de récursion
        self.tol = 40.0  # est réglable lors de l'execution
        self.step = 1  # est automatiquement réglé par le programme

        self.maxBrightness = 400
        self.minBrightness = 120
        self.marge = 0.5
        self.Qcoeff = (100, 20, 100, 20)
        self.view = 1

        # On définit la taille des indicateurs visuels / taille de l'image
        self.maxdist = int(video.Framessize[1] / 20)
        self.minsize = int(video.Framessize[1] / 100)
        self.crosswidth = int(video.Framessize[0] / 500)
        self.rectanglewidth = int(video.Framessize[1] / 1250)

        sys.setrecursionlimit(self.precision)
