# Projet-analyse-video
Cet algorithme a été développer dans le cadre de mon TIPE (épreuve faisant partie des concours d'entrée en école d'ingénieur).  
Je le considère terminé mais il peut certainement encore être amélioré.

## Objectif
L'objectif est de ce projet est de pouvoir effectuer des mesures par analyse vidéo.  
Les mesures en question consistes à suivre la position au cours du temps de repères placés sur l'objet que l'on veut étudier.  
Il peut être utiliser pour suivre des trajectoires ou la déformation d'un objet. Par exemple, dans le cadre de mon TIPE il m'a permis de mesurer la déformation d'une tour miniature lorsque soumise à un séisme. 

## Comment l'utiliser  
Il faut avoir intégrer à l'environnement les modules numpy, Opencv, pymediainfo et csv.  
Lorsque on exécute le programme dans la console de l'IDE un dosssier est créé sur le bureau de l'ordinateur dans lequel il faut glisser la vidéo à étudier (le programme accepte les vidéos au format mp4 ou mov les fichiers qui ne sont pas sous ses formats seront déplacés vers le bureau).
Pendant le traitement des informations sont transmises à l'utilisateur par le biais de la console.
À l'issue du traitement de la vidéo les résultats sont stockés dans un dossier sur le bureau. Il est possible de changer l'emplacement des dossiers utilisés dans le dossier Base.py.
Le programme de suivis étant basé sur une implémentation du filtre de Kallman(cf suite), il est nécessaire d'ajuster les coeficients pour obtenir un algorythme le plus performant possible.(Je n'est à ce jour pas trouvé de méthode pour cela)

## Principe de fonctionnement 
Cet algorithme est basé sur la couleur des repères visuels : tous les repères doivent être de la même couleur, à savoir rouges, verts ou bleus.
La video est dans un premier temps décomposée en un ensemble de frames puis on traite chacune de ces frames les unes après les autres.
Le traitement de la frame suit une approche de type parcours de graphe : on voit l'image comme une forêt où chacun des arbres est l'ensemble des pixels composants le contour de l'image d'un repère.  
On balaye la frame jusqu'à tomber sur un pixel au sein duquel le poids relatif de la couleur des repères est supérieur à un seuil fixé. On rejoins la bordure de l'image du repère puis on effectue un parcours en profondeur pour récupérer l'ensemble des pixels composants le contours du repère.
On en déduis la position du repère.  
Le suivis des repères d'une frame à la suivante est basé sur le principe du filtre de Kallman.  
De manière à ne pas avoir une complexité dépendante de la résolution de la vidéo étudiée, un pas dépendant de la taille des repères visuels est calculé durant la calibration.
