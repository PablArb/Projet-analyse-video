from Modules import *

from Paths import paths, add_subdata_dirs, delete_dir, create_dir
from Indicators import *

def resultsdownload(video, scale, crosswidth):
    reboot(video)
    videodownload()
    datadownload(video, scale)
    framesdownload(video, crosswidth)
    create_video(video, crosswidth)
    return None

def reboot(video):
    add_subdata_dirs(video.id)
    delete_dir('csv')
    delete_dir('frames')
    delete_dir('vidéodl')
    add_subdata_dirs(video.id)
    return None

def videodownload():
    create_dir('vidéodl')
    source = paths['video storage']
    destination = paths['vidéodl'] + '/vidéo' + '.mp4'
    sht.copy2(source, destination)
    os.remove(paths['video storage'])
    return None

def datadownload(video, scale):
    create_dir('csv')
    print('\nSauvegarde de la data en cours ...')
    nom_colonnes = ['frame', 'time']
    objects = []
    frames = video.frames
    for i in range (len(frames)):
        for obj in video.frames[i].identified_objects:
            if obj not in objects:
                objects.append(obj)
                nom_colonnes += ['X' + str(obj), 'Y' + str(obj)]
    dos = open(paths['csv'] + '/positions objets.csv', 'w')
    array = csv.DictWriter(dos, fieldnames=nom_colonnes)
    array.writeheader()
    for i in range (len(frames)):
        dico = {'frame': frames[i].id, 'time': round(int(frames[i].id.split('.')[1]) / video.Framerate, 3)}
        for obj in video.frames[i].identified_objects:
            dico['X' + str(obj)] = scale * video.frames[i].identified_objects[obj][0]
            dico['Y' + str(obj)] = scale * video.frames[i].identified_objects[obj][1]
        array.writerow(dico)
    dos.close()
    print('Sauvegarde de la data --------------------------------------------- OK')
    return None

def framesdownload(video, crosswidth):
    create_dir('non treated frames')
    create_dir('treated frames')
    print('\nSauvegarde des frames en cours ...')
    for frame in video.frames:
        name = paths['non treated frames'] + '/frame' + str(int(frame.id.split('.')[1])) + '.jpg'
        cv2.imwrite(name, frame.array)
        name = paths['treated frames'] + '/frame' + str(int(frame.id.split('.')[1])) + '.jpg'
        cv2.imwrite(name, cross_color(frame.array, frame.identified_objects, crosswidth))
    print('Sauvegarde des frames --------------------------------------------- OK')
    return None

def create_video(video, crosswidth):
    out = cv2.VideoWriter(paths['vidéodl'] + '/vidéo traitée' + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video.Framerate,
                          video.Framessize)
    print('\nSauvegarde de la vidéo en cours ...')
    for frame in video.frames:
        # img = np.uint8(cross_color(frame.array, frame.identified_objects))
        img = frame.array
        out.write(img)
    print('Sauvegarde de la vidéo -------------------------------------------- OK')
    return None
