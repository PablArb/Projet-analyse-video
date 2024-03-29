from Modules import sht, sys
from Base import Break
from Base import mess
from IHM import download, interact
from MainConstructor import Video
from Calibration import calibration, reboot
from VideoTreatment import videotreatment
from PyQt5.QtWidgets import QApplication
from TreatementSpecs import TreatementSpecs
from Gui_Main import CalibDisplay


def cleaner(video: Video, isOK=True) -> None:
    """
    video : Video, video que l'on souhaite traiter
    isOK : booléen, optionel, si le traitement n'est pas terminé, mais que l'utilisateur veut interompre l'algorithme,
        sa video sera copiée sur le bureau.

    Efface les traces que laisse l'algorithme sur loridnateur de l'utilisateur.
    """
    sys.setrecursionlimit(1000)
    if not isOK:
        dst = video.paths.desktop
        src = video.paths.videoStorage + '/' + video.id
        sht.copy2(src, dst)
    video.paths.delete_dir('videoStorage')
    video.paths.delete_dir('bac')
    return None


if __name__ == '__main__':
    try:
        treatementSpecs = TreatementSpecs()

        app = QApplication(sys.argv)

        print(mess.B_proc, end='')

        # On récupère la vidéo et ses caractéristiques
        video = Video(treatementSpecs)
        interact.setting_input(treatementSpecs.settings, 'lenref', 'float')

        # On traite la première frame pour vérifier que les réglages sont bons
        isOK = False
        while not isOK:
            # Tant que le traitement n'est pas satisfaisant on recommence cette étape
            CalibDisplay(video.Frames[0].array, treatementSpecs)
            calibration(video, treatementSpecs)
            if interact.yn(mess.I_val):
                isOK = True
            else:
                # lorsque le traitement n'est pas satisfaisant, il est proposé de modifier les paramètres.
                interact.verif_settings(video)
                reboot(video)
        
        # Une fois que tout est bon on traite la vidéo.
        videotreatment(video)
        
        # On télécharge les résultats.
        download.reboot(video)
        download.data(video)

        if interact.yn(mess.I_dlr):
            download.results(video)
        cleaner(video)
        print(mess.E_proc)

    except (KeyboardInterrupt, Break):
        cleaner(video, isOK=False)
        print(mess.E_proc)
