# Projet Python

*par Benjamin Cerf, Tristan Delhaye, et Mathieu Roig*

+ Pourquoi le Rugby ?

Le rugby, et en particulier le championnat français de rugby à XV du TOP14, offre une opportunité unique pour l’analyse de données sportives. Contrairement à des sports comme le football ou le basketball où les données sont déjà massivement exploitées, le rugby reste sous-analysé. 
Notre projet rend accessible les données du TOP14 et permet l'exploration et la modélisation des performances des équipes du championnat, le tout en espérant contribuer à une meilleure compréhension des facteurs clés influençant les résultats.


+ Nos Objectifs

Plusieurs sites comme [Opta Analyst](https://theanalyst.com/2024/01/club-rugby-stats-hub) ou [TOP14](https://top14.lnr.fr/calendrier-et-resultats) proposent un accès aux résultats du TOP14 et à des statistiques plus détaillées sur les joueurs ou les équipes. Notre démarche de les contacter pour obtenir des données n'ayant pas donné suite, nous avons cherché à rendre avant tout accessible et reproductible de telles données afin de pouvoir travailler dessus. Notre objectif de modélisation est quant à lui de prédire le classement et les résultats de la saison en cours. 

+ Sources des Données

Nos données sur les résultats des championnats de France de rugby à XV sont donc toutes récupérées par web-scraping à partir de leurs pages Wikipédia respectives. Bien que le format de ces pages puisse changer, le code devrait être reproductible sans trop de souci pour les années futures.


+ Présentation du Dépôt

Notre rendu consiste en un fichier ```main.ipynb```, aussi disponible avec les résultats pré-exécutés (```main_results.ipynb```). Afin de faciliter la compréhension de notre code mais aussi sa reproductibilité et son utilisation future, le dossier ```scripts``` contient la plupart de nos fonctions correspondant aux étapes principales du projet. Le dossier ```data``` contient une copie locale de nos données nettoyées, et enfin le fichier ```requirements.txt``` rappelle les installations nécessaires à l'exécution.


+ Licence

Ce projet est sous licence MIT.