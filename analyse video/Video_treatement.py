from Modules import *
from Paths import *
from Indicators import *

from Error_constructor import SettingError
from IHM import verif_settings

# Treatement tools

def videotreatement(video, maxdist, bordure_size):
    """
    Permet le traitement de l'ensemble des frames qui constituent la vidéo ainsi que le suivi des objets
    """
    global positions, definition, tol, c, minsize, crosswidth, rectanglewidth
    frames = video.frames
    obj_compteur = 0

    print('')

    # Initialisation
    for obj in positions[frames[0].id]:
        video.frames[0].identified_objects['obj-' + str(obj_compteur)] = positions[frames[0].id][obj]
        obj_compteur += 1

    bande1 = [i for i in range(0, bordure_size + 1)]
    bande2 = [i for i in range(video.Framessize[1] - bordure_size, video.Framessize[1] + 1)]

    for i in range(1, len(frames)):
        try :
            treated = frametreatement( frames[i].array )[0]
            positions[frames[i].id] = position(treated)

            for obj1 in positions[frames[i].id]:

                identified = False
                distances_list = {}
                x1, y1 = positions[frames[i].id][obj1][0], positions[frames[i].id][obj1][1]

                for obj2 in video.frames[i-1].identified_objects:
                    x2, y2 = video.frames[i-1].identified_objects[obj2][0], video.frames[i-1].identified_objects[obj2][1]
                    d = round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2), 2)
                    distances_list[obj2] = d

                if len(distances_list) != 0:
                    min_key = min(distances_list, key=distances_list.get)
                    distance = distances_list[min_key]
                    if distance < maxdist:
                        identified = True
                        video.frames[i].identified_objects[min_key] = positions[frames[i].id][obj1]

                if not identified:
                    if x1 in bande1 or x1 in bande2:
                        video.frames[i].identified_objects['obj-' + str(obj_compteur)] = [x1, y1]
                        obj_compteur += 1
                    if y1 in bande1 or y1 in bande2:
                        video.frames[i].identified_objects['obj-' + str(obj_compteur)] = [x1, y1]
                        obj_compteur += 1
        except SettingError :
            pass

        progression = round((int(frames[i].id.split('.')[1]) / (len(frames) - 1)) * 100, 1)
        print('\rTraitement de ' + video.id + ' en cours :', str(progression), '%', end='')
        t.sleep(.02)

    print('\nTraitement de ' + video.id + ' -------------------------------------------- Finit')
    return None

def frametreatement(frame):
    """
    Permet le traitement de la frame passée en argument.
    frame : tableau uint8.
    """
    global definition
    isOK = False
    while not isOK and definition <= 15:
        try:
            NB_im = prep(frame)
            extremas = objects_identification(NB_im)
            isOK = True
        except RecursionError:
            print('\rDéfinition trop élevée, tentative avec une défintion plus faible', end='')
            definition += 1
            frametreatement(frame)

    if isOK:
        extremas = rectifyer(extremas)
        return extremas, NB_im
    else:
        raise SettingError


# Frame manipulation tools

def get_neighbours(image, pixel: list) -> list:
    """
    Renvoie la liste des voisins du pixel 'pixel' à étudier dans le cadre de la recherche d'objet.
    image : image en N&B.
    pixel : sous la forme [j,i].
    """
    x, y = pixel[0], pixel[1]
    h = len(image)
    w = len(image[0])
    L_neighours_to_test = [[(x - 1) % w, (y - 1) % h], [(x - 1) % w, y], [(x - 1) % w, (y + 1) % h],
                           [x, (y - 1) % h], [x, (y + 1) % h],
                           [(x + 1) % w, (y - 1) % h], [(x + 1) % w, y], [(x + 1) % w, (y + 1) % h]]
    L_neighours = []
    for element in L_neighours_to_test:
        if image[element[1]][element[0]] == 255:
            L_neighours.append(element)
    return L_neighours

def visiter(image, depart: list, object: list, extr: list) -> list:
    """
    Regroupe tous les pixels appartenant a un même objets (forme blanche ici) sous la forme d'une liste.
    image : image en N&B.
    depart : pixel duquel on va partir pour 'explorer' notre objet, sous la forme [j,i].
    objet : liste contenant tout les pixels appartenants au même objet.
    """
    if depart not in object:
        object.append(depart)
        # xmin, ymin, xmax, ymax = extr[0], extr[1], extr[2], extr[3] (pour info)
        if depart[0] < extr[0]:
            extr[0] = depart[0]
        if depart[1] < extr[1]:
            extr[1] = depart[1]
        if depart[0] > extr[2]:
            extr[2] = depart[0]
        if depart[1] > extr[3]:
            extr[3] = depart[1]
    for pixel in get_neighbours(image, depart):
        if pixel not in object:
            visiter(image, pixel, object, extr)
    return object, extr

def discovery(image, depart: list) -> list:
    object = [depart]
    init_extr = [depart[0], depart[1], depart[0], depart[1]]
    infos = visiter(image, depart, object, init_extr)
    object = infos[0]
    extr = infos[1]
    return object, extr

def objects_identification(image) -> dict:
    """
    Regroupe tout les objets de l'image dans un dictionnaire.
    image : image en N&B sous la forme d'un array de 0 et 255.
    """
    h = len(image)
    w = len(image[0])
    objects = {}
    extremas = {}
    n = 0
    for j in range(h):
        for i in range(w):
            if image[j][i] == 255:
                element_in = False
                for obj in objects:
                    if [i, j] in objects[obj]:
                        element_in = True
                if not element_in:
                    infos = discovery(image, [i, j])
                    objects[n] = infos[0]
                    extremas[n] = infos[1]
                    n += 1
    for obj in extremas:
        xmin, ymin, xmax, ymax = extremas[obj][0], extremas[obj][1], extremas[obj][2], extremas[obj][3]
        extremas[obj] = [xmin * definition, ymin * definition, xmax * definition, ymax * definition]
    return extremas

def position(extremas: dict) -> list:
    """
    Récupère la position d'un objet à partir des extremas.
    Renvoie un dictionnaire où les clefs sont les noms des ifférents objets détectés sur la frame étudiée et les valeurs
    sont les coordonées du 'centre' de l'objet.
    """
    position = {}
    for obj in extremas:
        x = (extremas[obj][0] + extremas[obj][2]) / 2
        y = (extremas[obj][1] + extremas[obj][3]) / 2
        position[obj] = [x, y]
    return position

def rectifyer(extremas: dict) -> dict:
    """
    Rectifie quelques erreurs.
    """
    # On supprime les objets trop petits, probablement issus d'erreurs.
    global minsize
    problematic_objects = []
    for obj in extremas:
        if extremas[obj][2] - extremas[obj][0] < minsize or extremas[obj][3] - extremas[obj][1] < minsize:
            problematic_objects.append(obj)
    for obj in problematic_objects:
        del extremas[obj]
    return extremas


# Frame preparation tools

def rate_rgb(pixel: list) -> float:
    """
    Calcul le poids relatif de la composante c du pixel pixel parmis les composantes rgb qui le définissent.
    pixel : élement de l'image d'origine sous la forme [r, g, b].
    c = 0(rouge), 1(vert) ou 2(bleu).
    """
    global c
    assert c in [0, 1, 2]
    # la rédaction ci-dessous n'est pas idéale, mais l'utilisation du np.sum rend le traitement trop long
    return int(pixel[c]) / (int(pixel[0]) + int(pixel[1]) + int(pixel[2]) + 1)

def prep(image):
    """
    Renvoie une image en noir et blanc
    image : image de depart.
    Definition : l'image finale contiendra 1/definition² pixels de l'image initiale. Attention les dimensions de l'image
    sont donc modifiées.
    """
    global definition
    assert 0 < definition
    assert type(definition) == int
    simplified_im = []
    h = len(image)
    w = len(image[0])
    for i in range(int(h / definition)):
        line = []
        for j in range(int(w / definition)):
            pixel = image[i * definition][j * definition]
            if rate_rgb(pixel) < tol:
                line.append(0)
            else:
                line.append(255)
        simplified_im.append(line)
    return simplified_im


# Calibration fcts

def calibration(video, definition2, tol2, c2, minsize2, crosswidth2, rectanglewidth2):
    """
    À effectuer avant le traitement de l'ensemble de la vidéo pour vérifier le bon réglage de l'ensmeble des paramètres.
    """
    global positions, definition, tol, c, minsize, crosswidth, rectanglewidth
    definition, tol, c, minsize, crosswidth, rectanglewidth = definition2, tol2, c2, minsize2, crosswidth2,\
                                                              rectanglewidth2
    positions = {}

    print('\nTraitement en cours ...')
    first = copy_im(video.frames[0].array)

    try :
        detected = frametreatement(first)
    except SettingError :
        print('\nIl y a un problème, veuillez vérifiez les réglages')
        verif_settings()
        definition = 1
        calibration()
        return None

    extremas = detected[0]
    positions[video.frames[0].id] = position(rectifyer(detected[0]))

    print('\nTraitement -------------------------------------------------------- OK')

    images_names = []
    create_dir('calib')

    color_im = first
    images_names.append('color_im')
    fill_calibdir(color_im, 'color_im')

    NB_im = cv2.resize(np.uint8(detected[1]), video.Framessize)
    images_names.append('NB_im')
    fill_calibdir(NB_im, 'NB_im')

    treated_NB = np.uint8(rectangle_NB(NB_im, extremas, rectanglewidth))
    images_names.append('treated_NB')
    fill_calibdir(treated_NB, 'treated_NB')

    treated_color = np.uint8(cross_color(color_im, positions[video.frames[0].id], crosswidth))
    images_names.append('treated_color')
    fill_calibdir(treated_color, 'treated_color')

    print("\nAffichage du résultat, veuillez checker sa correction\n(une fenêtre a dû s'ouvrir)")
    calib_show(images_names)
    print('Validation du résultat -------------------------------------------- OK')

    sht.rmtree(paths['calib'])
    return None

def copy_im (image):
    L = len(image)
    l = len(image[0])
    newIm = []
    for y in range (L):
        newLine = []
        for x in range(l):
            newLine.append(image[y][x])
        newIm.append(newLine)
    return np.uint8(newIm)

def fill_calibdir(image, image_name):
    cv2.imwrite(paths['calib'] + '/' + image_name + '.jpg', image)
    return None

def calib_show(images_names: list):
    for i in range(len(images_names)):
        cv2.imshow('Config Window - ' + images_names[i], cv2.imread(paths['calib'] + '/' + images_names[i] + '.jpg'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return None
