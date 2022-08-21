from Modules import *

user = gp.getuser()
paths = {}
paths['bac'] = '/Users/' + user + '/Desktop/bac'
paths['calib'] = '/Users/' + user + '/Documents/##calibdir##'
paths['video storage'] = '/Users/' + user + '/Documents/temporary storage.mp4'
paths['data'] = '/Users/' + user + '/Desktop/data'

def add_subdata_dirs(video):
    paths['csv'] = paths['data'] + '/' + video + '/csv'
    paths['vidéodl'] = paths['data'] + '/' + video + '/vidéo'
    paths['frames'] = paths['data'] + '/' + video + '/frames'
    paths['treated frames'] = paths['frames'] + '/treated'
    paths['non treated frames'] = paths['frames'] + '/non treated'
    return None

def create_dir(dir: str):
    p = paths[dir]
    try:
        if not os.path.exists(p):
            os.makedirs(p)
    except OSError:
        print('Error: Creating directory of data')
    return None

def delete_dir(dir: str):
    p = paths[dir]
    try:
        if os.path.exists(p):
            sht.rmtree(p)
    except OSError:
        print('Error: Creating directory of data')
    return None
