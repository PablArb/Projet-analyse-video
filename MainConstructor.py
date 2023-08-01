from Modules import os, t, sht, mi, np, cv2
from Base import paths, mess
from SettingsConstructor import Settings
from KallmanFilterConstructor import KallmanFilter
from TreatementSpecs import Treatement_specs


class Video(object):

    def __init__(self, Tspecs: Treatement_specs):
        self.paths = paths

        self.id = None  # titre de la vidéo
        self.videoinput()

        self.Framerate = self.get_framerate()  # nombre de frame par seconde
        self.Framessize = self.get_framessize()  # taille des frames
        self.Frames = self.get_frames()  # liste contenant les frames de la vidéo

        self.scale = None  # rapport distance sur nombre de pixels
        self.markercount = 0  # nombre de repères détectés sur la vidéo
        self.markers = []
        self.computationDuration = None  # temps mis par l'algorythme pour effectuer le traitement

        self.detSettings(Tspecs)

        self.treatementEvents = ''

    def videoinput(self) -> None:
        """
        Récupère la vidéo auprès de l'utilisateur.
        """
        self.paths.create_dir('bac')
        acceptedFormats = ['mp4', 'mov', 'MOV']
        isempty = True
        print(mess.B_vi0, end='')
        while isempty:
            if len(os.listdir(self.paths.bac)) != 0:
                isempty = False
            t.sleep(0.5)
        _bac = os.listdir(self.paths.bac)
        ext = _bac[0].split('.')[1]
        if len(_bac) == 1 and ext in acceptedFormats:
            video = _bac[0]
            self.paths.videoinput = self.paths.bac + '/' + video
            self.paths.create_dir('videoStorage')
            sht.copy2(self.paths.videoinput, self.paths.videoStorage)
            self.id = str(video)
            self.paths.delete_dir('bac')
            return None
        elif len(_bac) == 1 and ext not in acceptedFormats:
            print(mess.P_vi1, end='')
            source = self.paths.bac + '/' + _bac[0]
            destination = self.paths.desktop + '/' + _bac[0]
            sht.copy2(source, destination)
            self.paths.delete_dir('bac')
            t.sleep(2)
            self.videoinput()
        elif len(_bac) > 1:
            print(mess.P_vi2, end='')
            self.paths.delete_dir('bac')
            t.sleep(2)
            self.videoinput()

    def get_framerate(self) -> float:
        """
        Renvoie le nombre de frames par secondes de la vidéo étudiée.
        """
        media_info = mi.MediaInfo.parse(self.paths.videoStorage + '/' + self.id)
        tracks = media_info.tracks
        for i in tracks:
            if i.track_type == 'Video':
                framerate = float(i.frame_rate)
        return framerate

    def get_framessize(self) -> tuple:
        """
        Renvoie un tuple de deux valeurs : la hauteur et largeur des frames de la vidéo.
        """
        media_info = mi.MediaInfo.parse(self.paths.videoStorage + '/' + self.id)
        video_tracks = media_info.video_tracks[0]
        w = int(video_tracks.sampled_width)
        h = int(video_tracks.sampled_height)
        framessize = (w, h)
        return framessize

    def get_frames(self) -> list:
        """
        Renvoie une liste contenant l'ensemble des frames (tableaux de type uint8) dans le même ordre que dans la vidéo
        étudiée.
        """
        frames = []
        cam = cv2.VideoCapture(self.paths.videoStorage + '/' + self.id)
        frame_number = 0
        print(mess.B_gfs, end='')
        while True:
            ret, frame = cam.read()
            if ret:
                frames.append(Frame(frame_number, np.array(frame)))
                frame_number += 1
            else:
                break
        cam.release()
        cv2.destroyAllWindows()
        print(mess.E_gfs, end='')
        return frames
    
    def detSettings(self, Tspecs):
        Tspecs.settings.maxdist = self.Framessize[1] // 20
        Tspecs.settings.minsize = self.Framessize[1] // 100
        Tspecs.settings.crosswidth = self.Framessize[0] // 500
        Tspecs.settings.rectanglewidth = self.Framessize[1] // 1250
        Tspecs.settings.pinnedMarkersIndicatorRadius = self.Framessize[1] // 120



class Frame(object):
    def __init__(self, id, array):
        self.id = id
        self.array = array
        self.NBarray = None
        self.mesures = []
        self.identifiedObjects = []

    def detNBArray(self, settings) -> None:

        im = self.array
        imHSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        hueMin, hueMax = settings.hueWindow
        hueFilt = (imHSV[:, :, 0] >= hueMin) * (imHSV[:, :, 0] <= hueMax)

        satMin, satMax = settings.satWindow
        satFilt = (imHSV[:, :, 1] >= satMin) * (imHSV[:, :, 1] <= satMax)

        valMin, valMax = settings.valWindow
        valFilt = (imHSV[:, :, 2] >= valMin) * (imHSV[:, :, 2] <= valMax)

        imNB = hueFilt * satFilt * valFilt
        self.NBarray = np.uint8(imNB)
        return None



class Object(object):
    def __init__(self, id, initpos, initframe, dt, Qcoeff):
        self.id = id
        self.lastupdate = 0
        self.lastknownpos = initpos
        self.predictions = {initframe: initpos}
        self.positions = {initframe: initpos}
        self.kf = KallmanFilter(dt, initpos, Qcoeff)
        self.status = 'hooked'



class Mesure(object):
    def __init__(self, id, extr, borders):
        self.id = id
        self.extremas = extr  # [xmin, ymin, xmax, ymax]
        self.borders = borders
        self.pos = self.detPos()  # [x, y]
        self.size = self.detSize()

    def detPos(self) -> tuple:
        """
        extremas : dictionaire contenant les coordonnées extremales des repères détectés sur une frame.

        Détermine la position d'un objet à partir des extremas.
        Renvoie un dictionnaire où les clefs sont les noms des différents objets détectés sur la frame étudiée et les
        valeurs sont les coordonées du 'centre' de l'objet.
        """
        obj = self.extremas
        x, y = (obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2
        return x, y

    def detSize(self) -> tuple:
        extr = self.extremas
        xsize = extr[2] - extr[0]
        ysize = extr[3] - extr[1]
        return xsize, ysize
