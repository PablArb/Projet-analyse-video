import sys

class Settings(object):
    def __init__(self, video):
        self.modifiables = ('markerscolor', 'orientation', 'lenref',
                            'cth', 'maxBrightness', 'minBrightness',
                            'hueWindow', 'satWindow',
                            'marge', 'view')

        self.Qcoeff = (100, 20, 100, 20)  # coefficients de filtre de Kallman

        # paramètres automatiquement réglé par le programme
        self.precision = 1000  # défini la taille maximale que peut prendre un repère
        self.maxPrec = 1e6  # plafond imposé à la limitte de récursion
        self.step = 1  # pas ave lequel on parcourt les frames

        # paramètres réglables lors de l'execution

        self.markerscolor = None  # couleur des repères visuels sur la video
        self.orientation = None  # orientation de la video (paysage ou portrait)
        self.lenref = None  # longueur de référence associée à la video

        self.cth = 40.0  # taux seuil pour la couleur
        self.maxBrightness = 500
        self.minBrightness = 150

        self.hueWindow = None
        self.satWindow = (25, 255)
        self.valWindow = (80, 150)

        self.marge = 1
        self.view = 1

        # On définit la taille des indicateurs visuels / taille de l'image
        self.maxdist = int(video.Framessize[1] / 20)
        self.minsize = int(video.Framessize[1] / 100)
        self.crosswidth = int(video.Framessize[0] / 500)
        self.rectanglewidth = int(video.Framessize[1] / 1250)

        sys.setrecursionlimit(self.precision)

    def detHueWindow(self) -> tuple:
        c = self.markerscolor
        if c == 0:
            self.hueWindow = [(100, 140)]
        elif c == 1:
            self.hueWindow = [(45, 80)]
        elif c == 2:
            self.hueWindow = [(0, 20), (160, 180)]
        return None
