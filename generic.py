
import cv2

paths           = ... # path permettant d'atteindre la video.
c               = ... # couleur des repères dont on étudie les positions.
definition      = 1   # quantifie la perte d'info.
tol             = 0.4 # fixe le seuil a partir du quel on peut considérer q'un pixel est de coueur étudiée.

minsize         = ... # fixe la taille minimum que peut avoir un objet.
maxdist         = ... # fixe la distance maximale que ne doit pas parcourir un objet entre deux frame pour continuer d'etre identifier comme un meme objet.
bordure_size    = ... # fixe la largeur de la bande autour de la video dans laquelle on considere qu'un objet qui y apparait n'un pas un bug mais un objet qui entre dans le cadre.




def get_frames () :
    '''
    Récupère l'ensembe des frames.
    Renvoie un dictionaire où les clés sont les numéros de frames et le valeurs des tableau de type uint8.
    '''
    global video, frames
    frames = {}
    cam = cv2.VideoCapture(paths)
    frame_number = 0
    print ('\nRécupération de la vidéo en cours ...')
    while(True):
        ret,frame = cam.read()
        if ret :
            frames['frame.' + str(frame_number)] = frame
            frame_number += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    print ('\rRécupération de la vidéo ------------------------------------------ OK')
    return None



# Frame preparation tools

def rate_rgb (pixel:list) -> float :
    '''
    Calcul le poids relatif de la composante c du pixel pixel parmis les composantes rgb qui le  définissent.
    pixel : élement de l'image d'origine sous la forme [r, g, b].
    c = 0(rouge), 1(vert) ou 2(bleu).
    '''
    global c
    assert c in [0,1,2]
    # la rédaction ci-dessous n'est pas idéale mais l'utilisation du np.sum rend le traitement trop long
    return int(pixel[c]) / (int(pixel[0]) + int(pixel[1]) + int(pixel[2]) + 1)

def prep (image) :
    '''
    Renvoie une image en noir et blanc
    image : image de depart.
    definition : l'image finale contiendra 1/definition² pixels de l'image initiale. Attention les dimensions de 'image sont donc modifiées.
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
            if rate_rgb(pixel) < tol :
                line.append(0)
            else:
                line.append(255)
        simplified_im.append(line)
    return simplified_im



# Treatement tools

def frametreatement (frame) :
    '''
    Permet le traitement de la frame passée en argument.
    frame : tableau uint8.
    '''
    global definition
    isOK = False
    while not isOK and definition <= 15 :
        try :
            NB_im = prep(frame)
            extremas = objects_identification(NB_im)
            isOK = True
        except RecursionError :
            print ('\rDéfinition trop élevée, tentative avec une défintion plus faible', end='')
            definition += 1
            frametreatement (frame)

    if isOK :
        extremas = rectifyer(extremas)
        return extremas, NB_im
    else :
        return 'TolError'

def videotreatement () :
    '''
    Permet le traitement de l'ensemble des frames qui constituent la vidéo ainsi que le suivi des objets
    '''
    global video, frames, positions, maxdist, bordure_size, tracked_objects
    # positions = {}
    tracked_objects = {}
    obj_compteur = 0
    frames_keys = list(frames.keys())

    print('')

    # Initialisation
    tracked_objects [frames_keys[0]] = {}
    for obj in positions[frames_keys[0]] :
        tracked_objects [frames_keys[0]]['obj-' + str(obj_compteur)] = positions[frames_keys[0]][obj]
        obj_compteur += 1

    for i in range (1,len(frames_keys)) :
        tracked_objects [frames_keys[i]] = {}

        treated = frametreatement( frames[ frames_keys[i] ] )[0]
        positions[ frames_keys[i] ] = position(treated)

        for obj1 in positions[ frames_keys[i] ] :

            identified = False
            distances_list = {}
            x1, y1 = positions[frames_keys[i]][obj1][0], positions[frames_keys[i]][obj1][1]

            for obj2 in tracked_objects[ frames_keys[i-1] ] :
                x2, y2 = tracked_objects[frames_keys[i-1]][obj2][0], tracked_objects[frames_keys[i-1]][obj2][1]
                d = round( ( (x1-x2)**2 + (y1-y2)**2  )**(1/2), 2)
                distances_list[obj2] = d

            if len(distances_list) != 0 :
                min_key = min(distances_list, key = distances_list.get)
                distance = distances_list[min_key]
                if distance < maxdist :
                    identified = True
                    tracked_objects [frames_keys[i]][min_key] = positions[frames_keys[i]][obj1]

            if not identified :
                if x1 in [x for x in range(0,bordure_size+1)] or x1 in [x for x in range(Framesize[1]-bordure_size, Framesize[1]+1)] :
                    tracked_objects [frames_keys[i]]['obj-' + str(obj_compteur)] = [x1, y1]
                    obj_compteur += 1
                if y1 in [y for y in range(0,bordure_size+1)] or y1 in [y for y in range(Framesize[0]-bordure_size, Framesize[0]+1)] :
                    tracked_objects [frames_keys[i]]['obj-' + str(obj_compteur)] = [x1, y1]
                    obj_compteur += 1

        progression = round( (int(frames_keys[i].split('.')[1])/(len(frames)-1))*100, 1)
        print('\rTraitement de la vidéo en cours :', str(progression), '%', end='')
        t.sleep (.02)

    print ('\nTraitement de la vidéo -------------------------------------------- Finit')
    return None



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

def visiter (image, depart:list, object:list, extr:list) -> list :
    '''
    Regroupe tous les pixels appartenant a un même objets (forme blanche ici) sous la forme d'une liste.
    image : image en N&B.
    depart : pixel duquel on va partir pour 'explorer' notre objet, sous la forme [j,i].
    objet : liste contenant tout les pixels appartenants au même objet.
    '''
    if depart not in object :
        object.append(depart)
        # xmin, ymin, xmax, ymax = extr[0], extr[1], extr[2], extr[3] (pour info)
        if depart[0] < extr[0] :
            extr[0] = depart[0]
        if depart[1] < extr[1] :
            extr[1] = depart[1]
        if depart[0] > extr[2] :
            extr[2] = depart[0]
        if depart[1] > extr[3] :
            extr[3] = depart[1]
    for pixel in get_neighbours(image, depart) :
        if pixel not in object :
            visiter(image, pixel, object, extr)
    return object, extr

def discovery (image, depart:list) -> list :
    object = [depart]
    init_extr = [depart[0], depart[1], depart[0], depart[1]]
    infos = visiter(image, depart, object, init_extr)
    object = infos[0]
    extr = infos[1]
    return object, extr

def objects_identification (image) -> dict :
    '''
    Regroupe tout les objets de l'image dans un dictionnaire.
    image : image en N&B sous la forme d'un array de 0 et 255.
    '''
    h = len(image)
    w = len(image[0])
    objects = {}
    extremas = {}
    n = 0
    for j in range (h) :
        for i in range (w) :
            if image[j][i] == 255 :
                element_in = False
                for obj in objects :
                    if [i,j] in objects[obj] :
                        element_in = True
                if not element_in :
                    infos = discovery(image, [i,j])
                    objects[n] = infos[0]
                    extremas[n] = infos[1]
                    n += 1
    for obj in extremas :
        xmin, ymin, xmax, ymax = extremas[obj][0], extremas[obj][1], extremas[obj][2], extremas[obj][3]
        extremas[obj] = [xmin*definition, ymin*definition, xmax*definition, ymax*definition]
    return extremas

def position (extremas:dict) -> list :
    '''
    Récupère la position d'un objet à partir des extremas.
    Renvoie un dictionnaire où les clefs sont les noms des ifférents objets détectés sur la frame étudiée et les valeurs sont les coordonées du 'centre' de l'objet.
    '''
    position = {}
    for obj in extremas :
        x = ( extremas[obj][0] + extremas[obj][2] )/2
        y = ( extremas[obj][1] + extremas[obj][3] )/2
        position[obj] = [x,y]
    return position

def rectifyer (extremas:dict) -> dict :
    '''
    Rectifie quelques erreurs.
    '''
    # On supprime les objets trop petits, probablement issus d'erreurs.
    global minsize
    problematic_objects = []
    for obj in extremas:
        if extremas[obj][2]-extremas[obj][0] < minsize or extremas[obj][3]-extremas[obj][1] < minsize :
            problematic_objects.append(obj)
    for obj in problematic_objects :
        del extremas[obj]
    # On renome nos objets.
    i = 0
    dico2 = {}
    for obj in extremas :
        dico2 ['obj-' + str(i)] = extremas[obj]
        i += 1
    return dico2