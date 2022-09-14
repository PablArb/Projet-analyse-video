from Modules import *

from Paths import paths, delete_dir
from Error_constructor import Break
from Video_constructor import get_mode


def videoinput():
    isempty = True
    print('\nPlacez la vidéo (.mp4) à étudier dans le bac sur votre bureau.')
    while isempty:
        if len(os.listdir(paths['bac'])) != 0:
            isempty = False
        t.sleep(0.5)
    bac = os.listdir(paths['bac'])
    if len(bac) == 1 and bac[0].split('.')[1] == 'mp4':
        video = bac[0].split('.')[0]
        paths['vidéoinput'] = paths['bac'] + '/' + video + '.mp4'
        sht.copy2(paths['vidéoinput'], paths['video storage'])
        return video
    elif len(bac) == 1 and bac[0].split('.')[1] != 'mp4':
        print('Veuillez fournir une vidéo au format mp4')
        delete_dir('bac')
        videoinput()
    elif len(bac) > 1:
        print("Veuillez ne placer qu'un document dans le bac")
        delete_dir('bac')
        videoinput()

def cinput():
    while True :
        c = input('\nCouleur des repères à étudier (1=bleu, 2=vert, 3=rouge) : ')
        if c in ['1', '2', '3', 'break']:
            if c in ['1', '2', '3']:
                c = int(c)-1
                return c
            else:
                raise Break
        else:
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def refinput ():
    while True:
        l = input('\nlongueur entre les deux premiers repères(cm) : ')
        try :
            if l == 'break':
                raise Break
            lenref = float(l)
            return lenref
        except ValueError :
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')

def verif_settings (video, tol, c, mode):
    while True :
        print('\n1 orientation de la vidéo :', ['landscape', 'portrait'][mode])
        print('2 couleur des repères :', ['bleue', 'verte', 'rouge'][c])
        print('3 tolérance : ', tol)
        which = input('quel réglage vous semble-t-il éroné (0=aucun, 1, 2, 3) ? ')
        if which in ['0', '1', '2', '3', 'pres', 'break']:
            if which == '0':
                return tol, c
            elif which == '1':
                get_mode(video)[1]
                return tol, c
            elif which == '2':
                c = cinput()
                return tol, c
            elif which == '3':
                tol += float(input('\nTolérance actuelle : ', tol, ', implémenter de : '))
                return tol, c
            elif which == 'pres':
                sys.setrecursionlimit(int(input('setrecursionlimit : ')))
                return tol, c
            else :
                raise Break
        else:
            print ('vous devez avoir fait une erreur, veuillez réessayer')

def yn(question):
    assert type(question) == str
    while True:
        yn = input('\n' + question + '\n[y]/n : ')
        if yn in ['y', '', 'n', 'break']:
            if yn == 'y' or yn == '':
                return True
            elif yn == 'n':
                return False
            else:
                raise Break
        else:
            print('Vous devez avoir fait une erreur, veuillez rééssayer.')
