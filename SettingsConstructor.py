import sys

class Settings(object):
    def __init__(self, video):
        self.modifiables = ('tol', 'maxBrightness', 'minBrightness', 'marge', 'view')

        self.Qcoeff = (100, 20, 100, 20)  # coefficients de filtre de Kallman

        # paramètres automatiquement réglé par le programme
        self.precision = 1000  # défini la taille maximale que peut prendre un repère
        self.maxPrec = 1e6  # plafond imposé à la limitte de récursion
        self.step = 1  # pas ave lequel on parcourt les frames

        # paramètres réglables lors de l'execution
        self.cth = 40.0  # taux seuil pour la couleur
        self.maxBrightness = 500
        self.minBrightness = 150
        self.marge = 1
        self.view = 1

        # On définit la taille des indicateurs visuels / taille de l'image
        self.maxdist = int(video.Framessize[1] / 20)
        self.minsize = int(video.Framessize[1] / 100)
        self.crosswidth = int(video.Framessize[0] / 500)
        self.rectanglewidth = int(video.Framessize[1] / 1250)

        sys.setrecursionlimit(self.precision)
