# Cet algorythme permet de localiser les élements de couleur rouge, vert ou bleu présents sur la photo mise en entrée, fonctionne bien avec le format mp4
# Test
# Atention pour reconnaitre un objet ce programme utilise notamment un algorythme récursif, rendant impossible l'analyse d'image où l'objet intéressant couvre une partie trop importante de l'image (dû a la limite de taille de la pile de récurtion).

# Import modules

import matplotlib.pyplot as plt
import cv2
import os
import csv
import sys
import shutil as sht
import numpy as np
import time as t


# Frame preparation tools

def reducer (image) :
    '''
    Renvoie l'image de depart à laquelle on a retiré de l'information pour simplifier son traitetement.
    image : image de depart.
    definition : l'image finale contiendra 1/definition² pixels de l'image initiale.
    '''

    global definition
    assert 0 < definition
    assert type(definition) == int
    simplified_im = []
    L = len(image)
    l = len(image[0])
    for i in range (int(L/definition)):
        line = []
        for j in range (int(l/definition)):
            line.append(image[i*definition][j*definition])
        simplified_im.append(line)
    return simplified_im


def rate_rgb (pixel:list) -> float :

    '''
    Calcul le poids relatif de la composante c du pixel pixel parmis les composantes rgb qui le  définissent.
    pixel : élement de l'image d'origine sous la forme [r, g, b].
    c = 0(rouge), 1(vert) ou 2(bleu).
    '''

    global c
    return int(pixel[c]) / (int(pixel[0])+int(pixel[1])+int(pixel[2])+1)



def im_filter (image) -> list :

    '''
    Renvoie une image en N&B.
    Associe la valeur 0 ou 1 à chaque pixel en fonction du poids de la composante c de ce pixel (> ou < à la tolérance).
    image : image de depart ;
    c : 0(rouge), 1(vert) ou 2(bleu) ;
    tol : valeur de reference ou tolérance.
    '''

    global c, tol
    assert c in [0,1,2]
    assert 0 < tol < 1
    new_im = []
    for line in image :
        new_line = []
        for pixel in line :
            t = rate_rgb(pixel)
            if t < tol :
                new_line.append(0)
            else:
                new_line.append(1)
        new_im.append(new_line)
    return new_im


# Frame manipulation tools

def recherche_voisins (image, pixel:list) -> list :

    '''
    Renvoie la liste des voisins du pixel 'pixel' à étudier dans le cadre de la recherche d'objet.
    image : image en N&B.
    pixel : sous la forme [j,i].
    '''

    y, x = pixel[0], pixel[1]
    L = len(image)
    l = len(image[0])
    L_voisins_to_test = [[(y-1)%L,(x-1)%l],[(y-1)%L,x],[(y-1)%L,(x+1)%l],
                         [ y,     (x-1)%l],            [ y,     (x+1)%l],
                         [(y+1)%L,(x-1)%l],[(y+1)%L,x],[(y+1)%L,(x+1)%l]]
    L_voisins = []
    for element in L_voisins_to_test :
        if image[element[0]][element[1]] == 1 :
            L_voisins.append(element)
    return L_voisins


def visiter (image, depart:list, objet:list) -> list :

    '''
    Regroupe tous les pixels appartenant a un même objets (forme blanche ici) sous la forme d'une liste.
    image : image en N&B.
    depart : pixel duquel on va partir pour 'explorer' notre objet, sous la forme [j,i].
    objet : liste contenant tout les pixels appartenants au même objet.
    '''

    if depart not in objet :
        objet.append(depart)
    for pixel in recherche_voisins(image, depart) :
        if pixel not in objet :
            visiter(image, pixel, objet)
    return objet


def parcours_graphe_profondeur (image, depart:list) -> list :
    objet = [depart]
    objet = visiter(image, depart, objet)
    return objet


def objects_identification (image) -> dict :

    '''
    Regroupe tout les objets de l'image dans un dictionnaire.
    image : image en N&B.
    '''

    L = len(image)
    l = len(image[0])
    objets = {}
    n = 0
    for j in range (L) :
        for i in range (l) :
            if image[j][i] == 1 :
                element_in = False
                for obj in objets :
                    if [j,i] in objets[obj] :
                        element_in = True
                if not element_in :
                    objets[n] = parcours_graphe_profondeur(image, [j,i])
                    n += 1
    return objets


def objects_field (dico_objets:dict) -> dict :

    '''
    Récupère les quatres extremités de chaque objet.
    '''

    extremas = {}
    for key in dico_objets :
        xmin, ymin, xmax, ymax = dico_objets[key][0][1], dico_objets[key][0][0],dico_objets[key][0][1],dico_objets[key][0][0]
        for i in range (len(dico_objets[key])) :
            pixel = dico_objets[key][i]
            if pixel[1] < xmin :
                xmin = pixel[1]
            if pixel[0] < ymin :
                ymin = pixel[0]
            if pixel[1] > xmax :
                xmax = pixel[1]
            if pixel[0] > ymax :
                ymax = pixel[0]
            extremas[key] = [xmin*definition, ymin*definition, xmax*definition, ymax*definition]
    return extremas


def position (extremas) :

    '''
    Récupère la position d'un objet à partir des extremas.
    '''

    position = {}
    for obj in extremas :
        x = ( extremas[obj][0] + extremas[obj][2] )/2
        y = ( extremas[obj][1] + extremas[obj][3] )/2
        position[obj] = [x,y]
    return position


def rectifyer (objets:dict) :

    '''
    Rectifie quelques erreurs.
    '''

    # On supprime les objets trop petits, probablement issus d'erreurs.
    global taille_min
    problematic_objects = []
    for key in objets:
        if len(objets[key]) < taille_min :
            problematic_objects.append(key)
    for key in problematic_objects :
        del objets[key]

    # On renome nos objets.
    i = 0
    dico2 = {}
    for key in objets :
        dico2 [i] = objets[key]
        i += 1

    return dico2


# Rectangles/cross drawing tools

def rectangle_NB (image, extremas) :
    global definition
    new_im = []
    for line in image :
        new_line = []
        for pixel in line :
            new_line.append(pixel)
        new_im.append(new_line)
    L = len(image)
    l = len(image[0])
    for key in extremas :
        xmin, ymin, xmax, ymax = int(extremas[key][0]/definition), int(extremas[key][1]/definition), int(extremas[key][2]/definition), int(extremas[key][3]/definition)
        for i in range (xmin-2,xmax+3):
            new_im[(ymin-2)%L][i%l], new_im[(ymax+2)%L][i%l] = 1, 1
        for j in range (ymin-2,ymax+3):
            new_im[j%L][(xmin-2)%l], new_im[j%L][(xmax+2)%l] = 1, 1
    return new_im


def rectangle_color (image, extremas) :
    global rectanglewidth
    new_im = []
    for line in image:
        new_line = []
        for pixel in line :
            new_line.append(pixel)
        new_im.append(new_line)
    L = len(new_im)
    l = len(new_im[0])
    for key in extremas.keys() :
        xmin, ymin, xmax, ymax = extremas[key][0], extremas[key][1], extremas[key][2], extremas[key][3]
        for i in range (xmin-rectanglewidth, xmax+rectanglewidth+1):
            for n in range (rectanglewidth+1):
                new_im[(ymin-n)%L][i%l], new_im[(ymax+n)%L][i%l] = [0, 255, 0], [0, 255, 0]
        for j in range (ymin-rectanglewidth, ymax+rectanglewidth+1):
            for n in range (rectanglewidth+1):
                new_im[j%L][(xmin-n)%l], new_im[j%L][(xmax+n)%l] = [0, 255, 0], [0, 255, 0]
    return new_im


def cross_color (image, positions) :
    global crosswidth
    new_im = []
    for line in image:
        new_line = []
        for pixel in line :
            new_line.append(pixel)
        new_im.append(new_line)
    L = len(image)
    l = len(image[0])
    for obj in positions :
        x = int(positions[obj][0])
        y = int(positions[obj][1])
        for i in range (x-crosswidth*10, x+crosswidth*10+1 ) :
            for n in range (y-int(crosswidth/2), y+int(crosswidth/2)+1):
                new_im[n%L][i%l] = [0, 255, 0]
        for j in range (y-crosswidth*10, y+crosswidth*10+1) :
            for n in range (x-int(crosswidth/2), x+int(crosswidth/2)+1):
                new_im[j%L][n%l] = [0, 255, 0]
    return new_im


# Occiliary tools

def BGR2RGB (image) :

    '''
    Lors de la lecture de l'image, sont utilisés deux outils :
    L'un en convention BGR (cv2), l'autre en convention RGB (matplotlib.pyplot).
    Cette fonction sert à la bonne compatibilité entre ces deux outils.
    '''

    for line in image :
        for pixel in line :
            pixel[0], pixel[2] = pixel[2], pixel[0]
    return image


# Treatement fcts

def getframes () :
    global video
    frames = {}
    cam = cv2.VideoCapture('/Users/pabloarb/Desktop/bac/' + video + '.mp4')
    currentframe = 0
    print ('\nRécupération des frames en cours ...')
    while(True):
        ret,frame = cam.read()
        if ret:
            frames[currentframe] = frame
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    print ('Récupération des frames ------------------------------------------- OK\n')
    return frames

def getfirstframe () :
    global video
    cam = cv2.VideoCapture("/Users/pabloarb/Desktop/bac/" + video + '.mp4')
    return cam.read()[1]

def frametreatement (frame) :
    global definition
    isOK = False

    while not isOK and definition <= 15 :
        try :
            NB_im = im_filter(reducer(frame))
            objets = objects_identification(NB_im)
            isOK = True
        except RecursionError :
            print ('\rDéfinition trop élevée, tentative avec une défintion plus faible', end='')
            t.sleep(0.2)
            definition += 1
            frametreatement (frame)

    if isOK :
        objets = rectifyer(objets)
        return objets, NB_im
    else :
        return 'TolError'

def videotreatement () :
    global video
    global positions
    currentframe = 0
    positions = {}
    frames = getframes()
    for frame in frames :
        treated = frametreatement(frames[frame])[0]
        positions[frame] = position( objects_field(treated))
        progression = round( (frame/(len(frames)-1))*100, 1)
        print('\rTraitement de la vidéo en cours :', str(progression), '%', end='' )
        t.sleep (.05)
    print ('\nTraitement de la video -------------------------------------------- Finit')
    return None


# Calibration fcts

def calibration () :
    global video
    first = getfirstframe()

    print ('\nTraitement en cours ...')
    treated = frametreatement( first )
    if treated == 'TolError' :
        print ('\nLa tolérance doit être mal réglée, vérifiez le réglage\n')
        return None
    print ('\nTraitement -------------------------------------------------------- OK')

    print ('\nAnalyse en cours ...')
    objets = treated[0]
    extremas = objects_field(objets)
    positions = position(extremas)
    print ('Analyse ----------------------------------------------------------- OK')

    first = BGR2RGB(first)
    NB_im = treated[1]
    treated_NB = rectangle_NB(NB_im, extremas)
    treated_color = cross_color(first, positions)

    print ('\nAffichage du résultat en cours ...')
    plt.close()
    plt.figure('calibration window')
    plt.subplot(1,4,1)
    plt.imshow(first)
    plt.title('original')
    plt.subplot(1,4,2)
    plt.imshow(NB_im, cmap = 'gray')
    plt.title('N&B')
    plt.subplot(1,4,3)
    plt.imshow(treated_NB, cmap = 'gray')
    plt.title('N&B rectangle')
    plt.subplot(1,4,4)
    plt.imshow(treated_color)
    plt.title('final')
    plt.pause(0.1)
    print ('Affichage du résultat --------------------------------------------- OK')

    return None


# Data download fcts

def create_subdata_folder (name) :
    global video
    folder = '/Users/pabloarb/Desktop/TIPE/data/' + video + '/' + name
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print ('Error: Creating directory of data')
    return folder

def videodownload () :
    global video
    folder = create_subdata_folder('vidéo')
    source = "/Users/pabloarb/Desktop/bac/" + video + '.mp4'
    destination = folder + '/vidéo' + '.mp4'
    sht.copy2(source, destination)
    return None

def datadownload () :
    global video, positions
    folder = create_subdata_folder('csv')
    keys = list(positions[0].keys())
    nom_colonnes = ['frame']
    for el in keys :
        nom_colonnes += ['X'+str(el), 'Y'+str(el)]
    dos = open(folder + '/positions objets.csv', 'w')
    obj = csv.DictWriter(dos, fieldnames=nom_colonnes)
    obj.writeheader()
    for frame in positions :
        dico = {'frame' : frame}
        for el in keys :
            dico['X'+str(el)] = positions[frame][el][0]
            dico['Y'+str(el)] = positions[frame][el][1]
        obj.writerow(dico)
    dos.close()
    return None

def framesdownload () :
    global video, positions
    create_subdata_folder('frames/non treated')
    create_subdata_folder('frames/treated')
    frames = getframes()

    for frame in frames :

        name = '/Users/pabloarb/Desktop/TIPE/data/' + video + '/frames/non treated/frame' + str(frame) +'.jpg'
        cv2.imwrite(name, frames[frame])
        name = '/Users/pabloarb/Desktop/TIPE/data/' + video + '/frames/treated/frame' + str(frame) +'.jpg'
        cv2.imwrite(name, np.float32(cross_color(frames[frame], positions[frame])))

        progression = round( (frame/(len(frames)-1))*100, 1)
        print('\rSauvegarde des frames en cours :', str(progression), '%', end='' )

    print ('\nSauvegarde des frames --------------------------------------------- Finit')


# IHM

def create_bac () :
    try:
        if not os.path.exists('/Users/pabloarb/Desktop/bac'):
            os.makedirs('/Users/pabloarb/Desktop/bac')
    except OSError:
        print ('Error: Creating directory of data')
    return None

def videoinput () :
    while True :
        print ('Placez la vidéo à étudier dans le bac sur votre bureau.')
        video = input('Nom de la vidéo : ')
        if os.path.exists('/Users/pabloarb/Desktop/bac/' + video + '.mp4') :
            return video
        else :
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def cinput () :
    while True :
        c = input('Couleur des repères à étudier (0=bleu, 1=vert, 2=rouge) : ')
        if c in ['0', '1', '2'] :
            return int(c)
        else :
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def yn_calib () :
    while True :
        yn = input ('\nLe traitement est-il bon ?\ny/n : ')
        if yn in ['y', 'n']:
            return yn
        else :
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def yn_framesdl () :
    while True :
        yn = input ("Télécharger l'ensemble des frames ?\ny/n : ")
        if yn in ['y', 'n']:
            return yn
        else :
            print('Vous devez avoir fait une erreur, veuillez rééssayer')


# Main

def main ():

    global video, c, definition, tol, taille_min, crosswidth, rectanglewidth

    # Réglages de rapidité/précision/sensibilité par défault.
    definition = 1
    tol = 0.4
    taille_min = 10
    # Largeur des bordures des rectangles/croix.
    crosswidth = 2
    rectanglewidth = 5

    print ('\nInitialisation de la procédure\n')

    create_bac()
    video = videoinput()

    print(' ')
    c = cinput()

    isOK = False
    while not isOK :
        calibration()
        yn = yn_calib()
        if yn == 'y':
            isOK = True
        else :
            i = input('\nTolérance actuelle : ' + str(tol) + ', implémenter de : ')
            tol += float(i)

    videotreatement()
    videodownload()
    datadownload ()

    print(' ')
    yn = yn_framesdl()
    if yn == 'y' :
        framesdownload()

    sht.rmtree('/Users/pabloarb/Desktop/bac')

    print ('\nProcédure terminée')

    return None


if __name__ == '__main__':
    sys.exit(main())