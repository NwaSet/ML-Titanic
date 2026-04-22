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
