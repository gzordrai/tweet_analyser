# pje

## Authors

Ali Ashraf
Thibault Tisserand

## Description générale du projet

### Description de la problématique

Ce projet à pour but concevoir une application capable de classifier le sentiment général (positif, négatif, neutre) exprimé dans des tweets portant sur un sujet spécifique. Pour atteindre cet objectif, plusieurs algorithmes d'apprentissage supervisé classiques, notamment le Dictionnaire, KNN et Bayes, seront mis en œuvre. Les performances de ces approches seront ensuite comparées et analysées afin d'identifier les plus adaptées à cette problématique.

### Description générale de l'architecture du projet

Le projet est structuré en trois "packages" principaux :

- `api` : Ce package sert d'interface entre l'application graphique et la logique de machine learning.
- `app` : Il correspond à l'interface graphique du projet, qui utilise l'API pour interagir avec le reste de l'application.
- `core` : Ce package contient toute la logique liée au machine learning et aux classifieurs.

L’API a été développée avec `FastAPI` et représente une petite partie du projet. L'application graphique, quant à elle, a été réalisée avec `PyQT6`, une bibliothèque Python dédiée à la création d’applications de bureau. Enfin, la partie `core` a été entièrement codée à la main en s’appuyant uniquement sur `numpy` pour garantir de bonnes performances et sur `tqdm` pour afficher une barre de progression lors des itérations.

La partie core utilise des principes de programmation orientée objet, comme le polymorphisme, ainsi que des design patterns tels que Strategy. Ce dernier a été mis en place pour anticiper l'ajout futur de nouveaux classifieurs, ce qui s'est avéré très utile pour intégrer le classifieur Bayesien par la suite.

[insert image of class diagram]

### Description de l'organisation au sein du binôme

Pour l'organisation dans le binôme, nous avons d'abord chacun travaillé sur le code des premières séances séparément, en utilisant des branches distinctes, ce qui nous a permis de pratiquer individuellement. Par la suite, nous avons regroupé tout le code pour avancer ensemble sur le projet. Ali s'est principalement occupé de la partie graphique tout en proposant des idées pour améliorer l'IA. Thibault, de son côté, a pris en charge la partie core, en développant des classes qui facilitent l'intégration avec l'API. Concernant l'API, nous avons travaillé ensemble dessus, car elle constitue le lien direct entre nos deux parties.

## Détails des différents travaux réalisés

### Préparation/nettoyage des données, base d'apprentissage

Done cf Ali's work

### Algorithmes de classification

Pour le projet, nous avons implémenté quatre algorithmes de classification. Le premier consistait à utiliser deux dictionnaires de mots : l'un contenant l'ensemble des mots positifs et l'autre l'ensemble des mots négatifs. Cette méthode, bien qu'introduite comme une approche de base, a montré ses limites pour la classification des tweets. Par conséquent, nous ne l'avons pas conservée dans notre application finale.

Nous avons ensuite implémenté le classifieur KNN (k-nearest neighbors), qui repose sur le calcul de la distance entre le tweet à classifier et tous les tweets de notre jeu d'entraînement. Une fois cette étape terminée, nous sélectionnons les k (un paramètre défini par l'utilisateur via notre interface graphique) voisins les plus proches, déjà classifiés, et comptons le nombre de voisins positifs, neutres ou négatifs. Enfin, en fonction des annotations de ces k voisins, nous déterminons l'annotation finale du tweet original.

Par la suite, nous avons implémenté un classifieur Bayesien en utilisant deux approches principales pour représenter les tweets. Dans un premier temps, nous avons opté pour une représentation par présence : chaque mot du vocabulaire est traité comme un attribut booléen indiquant sa présence ou son absence dans un tweet. Nous avons ignoré des aspects comme l’ordre des mots, leur syntaxe ou leur fréquence, ce qui simplifie considérablement le modèle. En appliquant le théorème de Bayes, nous avons calculé la probabilité qu’un tweet appartienne à une classe donnée (positive, neutre ou négative) en combinant les probabilités des mots présents dans le tweet pour chaque classe. La classe ayant la probabilité la plus élevée était attribuée au tweet.

Pour améliorer ce modèle, nous avons implémenté une variante basée sur la représentation par fréquence. Cette approche prend en compte non seulement la présence d’un mot, mais également le nombre de fois qu’il apparaît dans le tweet (sac de mots). Cela a nécessité une modification de l’équation pour inclure la fréquence de chaque mot, tout en conservant la logique générale du modèle Bayesien. Enfin, pour optimiser le classifieur, nous avons filtré les mots non significatifs, tels que les articles ou pronoms de moins de trois lettres, et expérimenté avec des n-grammes (bi-grammes et combinaisons uni-grammes + bi-grammes) afin de capturer des relations entre mots et d’améliorer les performances globales.

### Interface graphique

Done cf Ali's work

## Résultats de la classification avec les différentes méthodes et analyse

...

## Conclusions

...
