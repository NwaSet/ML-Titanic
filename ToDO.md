# Correction remise 2 :

## A Ajouter :
**faire le F1 score** car celui-ci permet de reprendre le taux de _bon positif_. \
Pour nous, on ne peut pas juste nous baser sur ``l'accuracy`` car celui-ci peux prédire plus de mort, alors que sur notre dataSet, on a beaucoup plus de mort qu'autre chose. \
si _dataSet_ **pas uniforme** -> **F1 Score**

Exemple :
```python
from sklearn.metrics import f1_score

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

score = f1_score(y_true, y_pred)
print(score)
```

## La Suite :

1. Faire La Fin De Titanic = 
    - correction de F1 Score 
    - le néttoyage des données dans le PréPrecessing et pas dans l'analyse descriptive.
    - ajout d'un pipeline pour automatiser le tout
    - faire un beau fichier xls pour une analyse des données (pas csv, ou alors faire la ligne de début du csv= nom des colonnes).

## Comment faire : 
- tkinter glisser déposer puis tout automatique. 
- deux frames 
    - glisser déposée
    - analyses des résultats, choix meilleur model + demander le résultats d'une nouvelle ligne.

# urgent !!!!!!!!!
corriger le préprocessing et mettre le sex dans catégoriel !!!!!!!!!!!!!!!!!!!!!!!
