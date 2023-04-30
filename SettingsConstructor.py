import sys

class Settings(object):
    def __init__(self, video):
        self.modifiables = {
                            'orientation': 'int',
                            'lenref': 'float', 'marge': 'int', 'view': 'int',
                            'hueWindow': 'tuple', 'satWindow': 'tuple', 'valWindow': 'tuple'
                            }

        # paramètres automatiquement réglé par le programme
        self.precision = 1000  # défini la taille maximale que peut prendre un repère
        self.maxPrec = 1e6  # plafond imposé à la limitte de récursion
        self.step = 1  # pas ave lequel on parcourt les frames

        # paramètres renseignés par l'utilisateur
        self.orientation = None  # orientation de la video (paysage ou portrait)
        self.lenref = None  # longueur de référence associée à la video

        # paramètres réglés par défaut, peuvent être ajusté durant l'éxécution
        self.hueWindow = (45, 80)
        self.satWindow = (35, 255)
        self.valWindow = (80, 150)
        self.marge = 1  #
        self.view = 1  # rayon du champ de vision de l'algorythme de parcours de graphe en nb de pixel

        self.Qcoeff = (100, 5, 100, 5)  # coefficients de filtre de Kallman

        # On définit la taille des indicateurs visuels / taille de l'image
        self.maxdist = int(video.Framessize[1] / 20)
        self.minsize = int(video.Framessize[1] / 100)
        self.crosswidth = int(video.Framessize[0] / 500)
        self.rectanglewidth = int(video.Framessize[1] / 1250)

        sys.setrecursionlimit(self.precision)
