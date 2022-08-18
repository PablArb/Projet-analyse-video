# Projet-analyse-video

Cet algorythme permet de localiser les repères de couleur rouge, vert ou bleu présents sur la vidéo mise en entrée, fonctionne avec le format mp4.
Fonctionne sur mac avec python v3.9 avec les modules pymediainfo, numpy et cv2 installés.

# Listes des variables

c : (entier) couleur des repères étudiés.

positions : (dictionaire dont chaque clé correspond à une frame et la valeure associée est un dictionairecomprends les positions des objets détectés) positions des repères détectés à chacune des frames de la vidéo.

tracked_objects : idem que positions seulement désormais les objets sont suivis et non plus identifiés par leurordre de découverte.