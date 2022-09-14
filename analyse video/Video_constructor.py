from Modules import *

from Paths import paths
from Error_constructor import Break


class Video:
    def __init__(self, id):
        self.id = id
        self.frames = get_frames(self)
        self.Framerate = get_framerate(self)
        self.Framessize = get_framessize()
        self.mode = None
        self.scale = None


class Frame:
    def __init__(self, id, array):
        self.id = id
        self.array = array
        self.identified_objects = {}

def get_framerate(video):
    """
    Renvoie dans le spectre global un dictionaire avec en clefs les numéros des frames et en valeurs des tableaux de
    type uint8.
    """
    media_info = mi.MediaInfo.parse(paths['vidéoinput'])
    tracks = media_info.tracks
    for i in tracks:
        if i.track_type == 'Video':
            Framerate = float(i.frame_rate)
    video.Framerate = Framerate
    return Framerate

def get_framessize():
    """
    Renvoie dans le spectre global un tuple de deux valeurs : la hauteur et largeur des frames de la video.
    """
    media_info = mi.MediaInfo.parse(paths['vidéoinput'])
    video_tracks = media_info.video_tracks[0]
    Framessize = [int(video_tracks.sampled_width), int(video_tracks.sampled_height)]
    return Framessize

def get_mode(video ,Framessize):
    # Framessize = get_framessize()
    while True:
        mode = input('\nLa vidéo est en mode (1=landscape, 2=portrait) : ')
        if mode in ['1', '2', 'break']:
            if mode == '1':
                height = min(Framessize)
                width = max(Framessize)
            elif mode == '2':
                height = max(Framessize)
                width = min(Framessize)
            else:
                raise Break
            Framessize = (width, height)
            video.Framessize = Framessize
            video.mode = int(mode)
            return Framessize, int(mode)
        else:
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')
            get_mode(video)

def get_frames(video):
    """
    Récupère l'ensembe des frames.
    Renvoie un dictionaire où les clés sont les numéros de frames et les valeurs des tableaux de type uint8.
    """
    frames = []
    cam = cv2.VideoCapture(paths['vidéoinput'])
    frame_number = 0
    print('\nRécupération de la vidéo en cours ...')
    while True:
        ret, frame = cam.read()
        if ret:
            frames.append(Frame('frame.' + str(frame_number), frame))
            frame_number += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    print('\rRécupération de la vidéo ------------------------------------------ OK')
    video.frames = frames
    return frames

def detScale (video, positions:dict, lenref):
    a = list(positions.keys())[0]
    b = list(positions.keys())[1]
    apos, bpos = positions[a], positions[b]
    xa , ya , xb, yb = apos[0], apos[1], bpos[0], bpos[1]
    scale = lenref / ( ( (xa-xb)**2 + (ya-yb)**2 )**(1/2) )
    video.scale = scale
    return scale
