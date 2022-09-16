import os

from Modules import *

user = gp.getuser()

paths = {}
L_paths = ['bac', 'calib', 'video storage', 'data']
paths[L_paths[0]] = '/Users/' + user + '/Desktop/bac'
paths[L_paths[1]] = '/Users/' + user + '/Desktop/##calibdir##'
paths[L_paths[2]] = '/Users/' + user + '/Desktop/temporary storage.mp4'
paths[L_paths[3]] = '/Users/' + user + '/Desktop/data'

if os.name == 'nt':
    for el in L_paths:
        paths[el] = 'C:'+paths[el]

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
