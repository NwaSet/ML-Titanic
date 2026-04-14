# Choix du meilleur modèle

## Réalisé par :

- [Samain Florian](https://github.com/NwaSet) — 2ème année, Intar D  
- [Ducourtieux Yohann](https://github.com/Nhanyo) — 2ème année, Intar D  

## Objectif :

L’objectif de cette section est de déterminer le meilleur modèle parmi l’ensemble de ceux testés lors des étapes précédentes.

## Choix effectué :

**Nous avons décidé de retenir :**  
- ``Decision Tree`` avec une sélection de variables basée sur le ``Variance Threshold``.

## Résultats obtenus :

- Accuracy sur le jeu de test : `0.8123`  
- Écart-type de la cross-validation : `0.0064`  

## Justification :

Malgré la présence d’un léger overfitting initial, ce modèle a été retenu après optimisation des hyperparamètres.  
Il présente le meilleur taux de précision parmi les modèles testés, tout en conservant une bonne capacité de généralisation.

Par ailleurs, l’écart-type très faible observé lors de la validation croisée indique une stabilité des performances. Les résultats obtenus varient peu d’un fold à l’autre, ce qui renforce la fiabilité du modèle.