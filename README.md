

# **OPTIMISATION D'IMPLANTATION DES ARTISANS**

Ce dépôt contient les scripts d'entraînement et d'inférence des modèles d'opportunité d'implantation et de continuité d'exploitation à
trois et cinq ans, utilisés dans l'outil d'optimisation d'implantation des artisans développé pour le CRMANA. 
Les scripts peuvent être utilisés en local sous réserve de disposer d'accès aux données.

## Requirements

Les librairies nécessaires sont indiquées dans les fichiers equirements.txt présents dans les deux dossiers associés à l'inférence et 
à l'entrainement

## Entrainement

Le script entrainement.py fait appel aux cinq modules :
  - modele_survie_3_ans_entrainement.py concerne l'apprentissage du modèle de classification associé à la continuité d'exploitation à 3 ans avec l'évaluation des performances du modèle.
  La variable à inférer est un boolean de continuité d'exploitation à 3 ans.
  - modele_survie_5_ans_entrainement.py concerne l'apprentissage du modèle de classification associé à la continuité d'exploitation à 5 ans avec l'évaluation des performances du modèle.
  La variable à inférer est un boolean de continuité d'exploitation à 5 ans.
  - modeles_survie_shapley_values concerne le calcul des valeurs SHAP pour les modèles de continuité d'exploitation à 3 et 5 ans.
  - modele_opportunite_entrainement.py concerne l'apprentissage du modèle de régression du nombre d'établissments par maille IRIS associé à la modélisation
  de l'opportunité d'implantation dans une zone pour l'artisan.
  - modele_opportunite_shapley_values.py concerne le calcul des valeurs SHAP pour le modèle d'opportunité d'implantation.
  
Les paramètres généraux sont le métier de l'artisan.

## Inférence

Le script inferene.py permet de réaliser l'inférence sur l'ensemble des carreaux géographiques de la commune et des communes avoisinantes
associées à un projet artisanal et plus spécifiquement sur la zone envisagée par l'artisan.

