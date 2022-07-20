# Programme permettant de récupérer la position sur l'image de repères

import cv2

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
    assert c in [0,1,2]
    return int(pixel[c]) / (int(pixel[0])+int(pixel[1])+int(pixel[2])+1)



def im_filter (image) -> list :

    '''
    Renvoie une image en N&B.
    Associe la valeur 0 ou 1 à chaque pixel en fonction du poids de la composante c de ce pixel (> ou < à la tolérance).
    image : image de depart ;
    c : 0(rouge), 1(vert) ou 2(bleu) ;
    tol : valeur de reference ou tolérance.
    '''

    global tol
    assert 0 < tol < 1
    new_im = []
    for line in image :
        new_line = []
        for pixel in line :
            t = rate_rgb(pixel)
            if t < tol :
                new_line.append(0)
            else:
                new_line.append(255)
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
        if image[element[0]][element[1]] == 255 :
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
            if image[j][i] == 255 :
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
    global minsize
    problematic_objects = []
    for key in objets:
        if len(objets[key]) < minsize :
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


# video treatement

def getframes (videopath) :
    global frames
    frames = {}
    cam = cv2.VideoCapture(videopath)
    currentframe = 0
    while(True):
        ret,frame = cam.read()
        if ret:
            frames[currentframe] = frame
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    return None

def frametreatement (frame) :
    global definition
    isOK = False

    while not isOK and definition <= 15 :
        try :
            NB_im = im_filter(reducer(frame))
            objets = objects_identification(NB_im)
            isOK = True
        except RecursionError :
            definition += 1
            frametreatement(frame)

    if isOK :
        objets = rectifyer(objets)
        return objets, NB_im
    else :
        return 'TolError'

def videotreatement () :
    global video, frames, positions
    currentframe = 0
    positions = {}
    for frame in frames :
        treated = frametreatement(frames[frame])[0]
        positions[frame] = position( objects_field(treated))
    return None


# Main
# videopath : path de la video sur l'appareil
# color : couleur des repères (0=bleu, 1=vert, 2=rouge)

def main (videopath, color) :
    global c, definition, tol, minsize, crosswidth, rectanglewidth

    c = color
    # Réglages de rapidité/précision/sensibilité par défault.
    definition = 1
    tol = 0.4
    minsize = 10
    # Largeur des bordures des rectangles/croix.
    crosswidth = 2
    rectanglewidth = 5

    print('Initialisation de la procédure')

    getframes(videopath)
    videotreatement()

    print ('\nProcédure terminée')

    return None