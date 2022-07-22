# Cet algorythme permet de localiser les repères de couleur rouge, vert ou bleu présents sur la vidéo mise en entrée, fonctionne avec le format mp4.

# Foncyion sur mac avec python v3.9 avec les modules pymediainfo, numpy et cv2 innstallés


# Import modules

import pymediainfo as mi
import numpy as np
import cv2
import csv
import os
import sys
import shutil as sht
import time as t


# Frame preparation tools

def rate_rgb (pixel:list) -> float :
    '''
    Calcul le poids relatif de la composante c du pixel pixel parmis les composantes rgb qui le  définissent.
    pixel : élement de l'image d'origine sous la forme [r, g, b].
    c = 0(rouge), 1(vert) ou 2(bleu).
    '''
    global c
    assert c in [0,1,2]
    return int(pixel[c]) / (int(pixel[0])+int(pixel[1])+int(pixel[2])+1)

def prep (image) :
    '''
    Renvoie une image en noir et blanc
    image : image de depart.
    definition : l'image finale contiendra 1/definition² pixels de l'image initiale.
    '''
    global definition
    assert 0 < definition
    assert type(definition) == int
    simplified_im = []
    h = len(image)
    w = len(image[0])
    for i in range (int(h/definition)):
        line = []
        for j in range (int(w/definition)):
            pixel = image[i*definition][j*definition]
            t = rate_rgb(pixel)
            if t < tol :
                line.append(0)
            else:
                line.append(255)
        simplified_im.append(line)
    return simplified_im


# Frame manipulation tools

def get_neighbours (image, pixel:list) -> list :
    '''
    Renvoie la liste des voisins du pixel 'pixel' à étudier dans le cadre de la recherche d'objet.
    image : image en N&B.
    pixel : sous la forme [j,i].
    '''
    x, y = pixel[0], pixel[1]
    h = len(image)
    w = len(image[0])
    L_neighours_to_test = [[(x-1)%w,(y-1)%h],[(x-1)%w,y],[(x-1)%w,(y+1)%h],
                           [ x,     (y-1)%h],            [ x,     (y+1)%h],
                           [(x+1)%w,(y-1)%h],[(x+1)%w,y],[(x+1)%w,(y+1)%h]]
    L_neighours = []
    for element in L_neighours_to_test :
        if image[element[1]][element[0]] == 255 :
            L_neighours.append(element)
    return L_neighours


def visiter (image, depart:list, object:list) -> list :
    '''
    Regroupe tous les pixels appartenant a un même objets (forme blanche ici) sous la forme d'une liste.
    image : image en N&B.
    depart : pixel duquel on va partir pour 'explorer' notre objet, sous la forme [j,i].
    objet : liste contenant tout les pixels appartenants au même objet.
    '''
    if depart not in object :
        object.append(depart)

    for pixel in get_neighbours(image, depart) :
        if pixel not in object :
            visiter(image, pixel, object)
    return object


def discovery (image, depart:list) -> list :
    object = [depart]
    object = visiter(image, depart, object)
    return object


def objects_identification (image) -> dict :
    '''
    Regroupe tout les objets de l'image dans un dictionnaire.
    image : image en N&B.
    '''
    h = len(image)
    w = len(image[0])
    objects = {}
    n = 0
    for j in range (h) :
        for i in range (w) :
            if image[j][i] == 255 :
                element_in = False
                for obj in objects :
                    if [i,j] in objects[obj] :
                        element_in = True
                if not element_in :
                    objects[n] = discovery(image, [i,j])
                    n += 1
    return objects


def objects_field (dico_objects:dict) -> dict :
    '''
    Récupère les quatres extremités de chaque objet.
    '''
    extremas = {}
    for obj in dico_objects :
        xmin, ymin, xmax, ymax = dico_objects[obj][0][0], dico_objects[obj][0][1],dico_objects[obj][0][0],dico_objects[obj][0][1]
        for pixel in dico_objects[obj] :
            if pixel[0] < xmin :
                xmin = pixel[0]
            if pixel[1] < ymin :
                ymin = pixel[1]
            if pixel[0] > xmax :
                xmax = pixel[0]
            if pixel[1] > ymax :
                ymax = pixel[1]
            extremas[obj] = [xmin*definition, ymin*definition, xmax*definition, ymax*definition]
    return extremas


def position (extremas:dict) -> list :
    '''
    Récupère la position d'un objet à partir des extremas.
    '''
    position = {}
    for obj in extremas :
        x = ( extremas[obj][0] + extremas[obj][2] )/2
        y = ( extremas[obj][1] + extremas[obj][3] )/2
        position[obj] = [x,y]
    return position


def rectifyer (objects:dict) -> dict :
    '''
    Rectifie quelques erreurs.
    '''
    # On supprime les objets trop petits, probablement issus d'erreurs.
    global minsize
    problematic_objects = []
    for obj in objects:
        if len(objects[obj]) < minsize :
            problematic_objects.append(obj)
    for obj in problematic_objects :
        del objects[obj]
    # On renome nos objets.
    i = 0
    dico2 = {}
    for obj in objects :
        dico2 [i] = objects[obj]
        i += 1

    return dico2


# Treatement fcts

def get_frames () :
    global video, frames
    frames = {}
    cam = cv2.VideoCapture(paths['vidéoinput'])
    currentframe = 0
    print ('Récupération des frames en cours ...')
    while(True):
        ret,frame = cam.read()
        if ret:
            frames[currentframe] = frame
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    print ('\rRécupération des frames ------------------------------------------- OK')
    return None

def get_framerate () :
    global video, Framerate
    media_info = mi.MediaInfo.parse(paths['vidéoinput'])
    tracks = media_info.tracks
    for i in tracks :
        if i.track_type == 'Video' :
            Framerate = float(i.frame_rate)
    return None

def get_framessize () :
    global video, Framesize
    media_info = mi.MediaInfo.parse(paths['vidéoinput'])
    video_tracks =  media_info.video_tracks[0]
    width, height = int(video_tracks.sampled_width), int(video_tracks.sampled_height)
    Framesize = (height, width)
    return None

def frametreatement (frame) :
    global definition
    isOK = False

    while not isOK and definition <= 15 :
        try :
            NB_im = prep(frame)
            objects = objects_identification(NB_im)
            isOK = True
        except RecursionError :
            print ('\rDéfinition trop élevée, tentative avec une défintion plus faible', end='')
            definition += 1
            frametreatement (frame)

    if isOK :
        objects = rectifyer(objects)
        return objects, NB_im
    else :
        return 'TolError'

def videotreatement () :
    global video, frames, positions
    currentframe = 0
    positions = {}
    for frame in frames :
        treated = frametreatement(frames[frame])[0]
        positions[frame] = position( objects_field(treated))
        progression = round( (frame/(len(frames)-1))*100, 1)
        print('\rTraitement de la vidéo en cours :', str(progression), '%', end='')
        t.sleep (.05)
    print ('\nTraitement de la vidéo -------------------------------------------- Finit')
    return None


# Rectangles/cross drawing tools

def rectangle_NB (image, extremas) :
    L = len(image)
    l = len(image[0])
    for key in extremas :
        xmin, ymin, xmax, ymax = int(extremas[key][0]), int(extremas[key][1]), int(extremas[key][2]), int(extremas[key][3])
        for i in range (xmin-2,xmax+3):
            image[(ymin-2)%L][i%l], image[(ymax+2)%L][i%l] = 255, 255
        for j in range (ymin-2,ymax+3):
            image[j%L][(xmin-2)%l], image[j%L][(xmax+2)%l] = 255, 255
    return image

def rectangle_color (image, extremas) :
    global rectanglewidth
    L = len(image)
    l = len(image[0])
    for key in extremas.keys() :
        xmin, ymin, xmax, ymax = extremas[key][0], extremas[key][1], extremas[key][2], extremas[key][3]
        for i in range (xmin-rectanglewidth, xmax+rectanglewidth+1):
            for n in range (rectanglewidth+1):
                image[(ymin-n)%L][i%l], image[(ymax+n)%L][i%l] = [0, 255, 0], [0, 255, 0]
        for j in range (ymin-rectanglewidth, ymax+rectanglewidth+1):
            for n in range (rectanglewidth+1):
                image[j%L][(xmin-n)%l], image[j%L][(xmax+n)%l] = [0, 255, 0], [0, 255, 0]
    return image


def cross_color (image, positions) :
    global crosswidth
    L = len(image)
    l = len(image[0])
    for obj in positions :
        x = int(positions[obj][0])
        y = int(positions[obj][1])
        for i in range (x-crosswidth*10, x+crosswidth*10+1 ) :
            for n in range (y-int(crosswidth/2), y+int(crosswidth/2)+1):
                image[n%L][i%l] = [0, 255, 0]
        for j in range (y-crosswidth*10, y+crosswidth*10+1) :
            for n in range (x-int(crosswidth/2), x+int(crosswidth/2)+1):
                image[j%L][n%l] = [0, 255, 0]
    return image


# Calibration fcts

def calibration () :
    global video, frames, Framesize
    print ('Traitement en cours ...')
    first = frames[0]
    treated = frametreatement( first )
    if treated == 'TolError' :
        print ('\nLa tolérance doit être mal réglée, vérifiez le réglage\n')
        return None
    print ('\nTraitement -------------------------------------------------------- OK\n')

    print ('Analyse en cours ...')
    objets = treated[0]
    extremas = objects_field(objets)
    positions = position(extremas)
    print ('Analyse ----------------------------------------------------------- OK')
    print(' ')

    images_names = []
    create_dir('config')

    color_im = first
    images_names.append('color_im')
    fill_configdir(color_im, 'color_im')

    NB_im = cv2.resize(np.uint8(treated[1]), Framesize)
    images_names.append('NB_im')
    fill_configdir(NB_im, 'NB_im')

    treated_NB = np.uint8(rectangle_NB(NB_im, extremas))
    images_names.append('treated_NB')
    fill_configdir(treated_NB, 'treated_NB')

    treated_color = np.uint8(cross_color(color_im, positions))
    images_names.append('treated_color')
    fill_configdir(treated_color, 'treated_color')

    print ('Affichage du résultat, veuillez checker sa correction')
    config_show (images_names)
    print ('Validation du résultat -------------------------------------------- OK')

    return None

def fill_configdir (image, image_name) :
    cv2.imwrite(paths['config'] + '/' + image_name + '.jpg', image)
    return None

def config_show (images_names:list) :
    for i in range(len(images_names)):
        cv2.imshow('Config Window - ' + images_names[i], cv2.imread(paths['config'] + '/' + images_names[i] + '.jpg'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    sht.rmtree(paths['config'])
    return None


# Data download fcts

def videodownload () :
    global video
    source = "/Users/pabloarb/Desktop/bac/" + video + '.mp4'
    destination = paths['vidéodl'] + '/vidéo' + '.mp4'
    sht.copy2(source, destination)
    return None

def datadownload () :
    global video, positions, Framerate
    create_dir('csv')
    nom_colonnes = ['frame', 'time']
    objects = []
    for frame in positions :
        for obj in positions[frame] :
            if obj not in objects :
                objects.append(obj)
                nom_colonnes += ['X'+str(obj), 'Y'+str(obj)]
    dos = open(paths['csv'] + '/positions objets.csv', 'w')
    array = csv.DictWriter(dos, fieldnames=nom_colonnes)
    array.writeheader()
    for frame in positions :
        dico = {'frame' : frame, 'time' : round(frame/Framerate, 3)}
        for obj in positions[frame] :
            dico['X'+str(obj)] = positions[frame][obj][0]
            dico['Y'+str(obj)] = positions[frame][obj][1]
        array.writerow(dico)
    dos.close()
    return None

def framesdownload () :
    global video, frames, positions
    create_dir('non treated frames')
    create_dir('treated frames')
    for frame in frames :
        name = paths['non treated frames'] + '/frame' + str(frame) +'.jpg'
        cv2.imwrite(name, frames[frame])
        name = paths['treated frames'] + '/frame' + str(frame) +'.jpg'
        cv2.imwrite(name, np.uint8(cross_color(frames[frame], positions[frame])))
        progression = round( (frame/(len(frames)-1))*100, 1)
        print('\rSauvegarde des frames en cours :', str(progression), '%', end='' )
    print ('\nSauvegarde des frames --------------------------------------------- Finit')
    return None

def create_video ():
    global frames, Framerate, Framesize
    out = cv2.VideoWriter(paths['vidéodl'] + '/vidéo traitée' + '.mp4',cv2.VideoWriter_fourcc(*'mp4v'), Framerate, Framesize)
    for frame in frames :
        img = np.uint8(cross_color(frames[frame], positions[frame]))
        out.write(img)
        progression = round( (frame/(len(frames)-1))*100, 1)
        print('\rSauvegarde de la vidéo en cours :', str(progression), '%', end='' )
    out.release()
    print ('\nSauvegarde de la vidéo -------------------------------------------- Finit')
    return None


# IHM

def videoinput () :
    global video
    create_dir('bac')
    isempty = True
    print ('Placez la vidéo (.mp4) à étudier dans le bac sur votre bureau.')
    while isempty :
        if len(os.listdir(paths['bac'])) != 0 :
            isempty = False
        t.sleep(0.5)
    if len(os.listdir(paths['bac'])) == 1 and os.listdir(paths['bac'])[0].split('.')[1] == 'mp4':
        video = os.listdir(paths['bac'])[0].split('.')[0]
        return None
    if len(os.listdir(paths['bac'])) == 1 and os.listdir(paths['bac'])[0].split('.')[1] != 'mp4':
        print('Veuillez fournir une vidéo au format mp4')
        delete_dir('bac')
        videoinput()
    if len(os.listdir(paths['bac'])) >= 1 :
        print ("Veuillez ne placer qu'un document dans le bac")
        delete_dir('bac')
        videoinput()

def cinput () :
    global c
    while True :
        c = input('Couleur des repères à étudier (0=bleu, 1=vert, 2=rouge) : ')
        if c in ['0', '1', '2'] :
            c = int(c)
            return None
        else :
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def yn_calib () :
    while True :
        yn = input ('\nLe traitement est-il bon ?\n[y]/n : ')
        if yn in ['y', '', 'n']:
            if yn == 'y' or yn == '' :
                return True
            else :
                return False
        else :
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def yn_framesdl () :
    while True :
        yn = input ("\nTélécharger l'ensemble des frames ?\n[y]/n : ")
        if yn in ['y', '', 'n']:
            if yn == 'y' or yn == '' :
                return True
            else :
                return False
        else :
            print('Vous devez avoir fait une erreur, veuillez rééssayer')


# paths gestion

paths = {}
paths['data'] = 'Desktop/data'
paths['bac'] = 'Desktop/bac'
paths['config'] = 'Desktop/##configdir##'

def add_subdata_dirs ():
    global video
    paths['vidéoinput'] = paths['bac'] + '/' + video + '.mp4'
    paths['csv'] = paths['data'] + '/' + video + '/csv'
    paths['vidéodl'] = paths['data'] + '/' + video + '/vidéo'
    paths['frames'] = paths['data'] + '/' + video + '/frames'
    paths['treated frames'] = paths['frames'] + '/treated'
    paths['non treated frames'] = paths['frames'] + '/non treated'

def create_dir (dir:str) :
    p = paths[dir]
    try:
        if not os.path.exists(p) :
            os.makedirs(p)
    except OSError:
        print ('Error: Creating directory of data')
    return None

def delete_dir (dir:str) :
    sht.rmtree(paths[dir])
    return None


# Main

def main ():

    global definition, tol, minsize, crosswidth, rectanglewidth

    # Réglages de rapidité/précision/sensibilité par défault.
    definition = 1
    tol = 0.4
    minsize = 10
    # Largeur des bordures des rectangles/croix.
    crosswidth = 2
    rectanglewidth = 5

    print ('\nInitialisation de la procédure\n')

    videoinput()
    print(' ')

    add_subdata_dirs()
    get_framerate()
    get_frames()
    get_framessize()

    create_dir('vidéodl')
    videodownload()

    delete_dir('bac')
    print(' ')

    cinput()
    print(' ')

    isOK = False
    while not isOK :
        calibration()
        if yn_calib() :
            isOK = True
        else :
            i = input('\nTolérance actuelle : ' + str(tol) + ', implémenter de : ')
            tol += float(i)
    print(' ')

    videotreatement()
    print(' ')

    datadownload ()
    create_video()
    print(' ')

    if yn_framesdl() :
        print(' ')
        framesdownload()

    print ('\nProcédure terminée')

    return None

# execution

if __name__ == '__main__':
    sys.exit(main())