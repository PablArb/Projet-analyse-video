# Projet-analyse-video
Cet algorithme a été développer dans le cadre de mon TIPE (épreuve faisant oartie des concours d'entré en école d'ingénieur).
Je le considère terminé mais il peut certainement encore être améliorer.

## Objectif
L'objectif est de ce projet est de pouvoir effectuer des mesures par analyse vidéo. 
Les mesures en question consistes à suivre la position au cours du temps de repères placés sur l'objet que l'on veut étudier.
Il peut être utiliser pour suivre des trajectoires ou la déformation d'un objet. Par exemple, dans le cadre de mon TIPE il m'a permis de mesurer la déformation d'une tour miniature lorsque soumise à un séisme. 

## Comment l'utiliser
L'ensemble de ce projet est basé sur le language de programmation Python, il faut donc un IDE adapté.
Il faut de plus avoir intégrer à l'environnement les modules numpy, Opencv, pymediainfo et csv.


## Principe de fonctionnement 
Cet algorithme est basé sur la couleur des repères : tous les repères doivent être de la même couleur, à savoir rouges, verts ou bleus. 
La video est dans un premier temps décomposée en un ensemble de frames puis on traite chacune de ces frames les unes après les autres.
On balaye chaque frame jusqu'à tomber sur un pixel au sein duquel le poids relatif de la couleur des repères est supérieur à un seuil fixé.
On considère alors que le pixel est sur l'image du repère et on detecte à partir de la le contour de notre repère tout en recupérant les 
