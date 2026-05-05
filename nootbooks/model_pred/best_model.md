# Choix du meilleur modele

## Realise par :

- [Samain Florian](https://github.com/NwaSet) - 2eme annee, Intar D
- [Ducourtieux Yohann](https://github.com/Nhanyo) - 2eme annee, Intar D

## Objectif

L'objectif de cette section est de determiner le meilleur modele parmi l'ensemble des combinaisons testees.

Les modeles compares sont :

- Decision Tree
- KNN
- Random Forest

Chaque modele a ete teste avec plusieurs selections de variables :

- `VT` : Variance Threshold
- `P` : selection personnelle
- `KB` : KBest

## Metrique principale

La metrique principale retenue est le `F1-score`.

L'accuracy reste utile comme indicateur secondaire, mais elle ne suffit pas pour choisir le meilleur modele. Dans ce projet, il est important de tenir compte a la fois :

- de la precision, c'est-a-dire la qualite des predictions positives ;
- du recall, c'est-a-dire la capacite du modele a retrouver les survivants ;
- du F1-score, qui combine precision et recall.

## Choix effectue

**Nous avons decide de retenir :**

- `Random Forest` avec une selection de variables basee sur `Variance Threshold`.

Cette combinaison correspond a `RF + VT`.

## Resultats obtenus

Pour le modele `RF + VT`, les resultats sont :

- Accuracy train : `0.9001`
- Accuracy test : `0.8199`
- Accuracy moyenne en validation croisee : `0.8021`
- Ecart-type de l'accuracy en validation croisee : `0.0251`
- F1-score moyen en validation croisee : `0.7243`
- Ecart-type du F1-score en validation croisee : `0.0334`
- F1-score test : `0.7565`
- Precision : `0.7766`
- Recall : `0.7374`
- Custom accuracy : `0.9195`

Les meilleurs hyperparametres trouves sont :

```python
{
    "max_depth": 15,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "n_estimators": 200
}
```

## Justification

Le modele `RF + VT` est retenu car il obtient le meilleur F1-score parmi toutes les combinaisons testees.

Il obtient egalement la meilleure accuracy sur le jeu de test (`0.8199`) et le meilleur recall (`0.7374`). Cela signifie qu'il est non seulement performant globalement, mais qu'il retrouve aussi mieux les survivants que les autres modeles.

Le modele presente un ecart entre le train et le test (`0.9001` contre `0.8199`). Cet ecart montre un overfitting modere, mais il reste acceptable car les performances sur le jeu de test et en validation croisee restent les meilleures du projet.

## Comparaison avec les autres modeles

Les meilleurs scores F1 observes sont :

| Selection | Modele | Accuracy test | F1-score | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| VT | RF | 0.8199 | 0.7565 | 0.7766 | 0.7374 |
| VT | DT | 0.8123 | 0.7461 | 0.7660 | 0.7273 |
| P | DT | 0.8046 | 0.7385 | 0.7500 | 0.7273 |
| KB | DT | 0.8008 | 0.7234 | 0.7640 | 0.6869 |
| P | RF | 0.7931 | 0.7128 | 0.7528 | 0.6768 |
| KB | KNN | 0.7931 | 0.7065 | 0.7647 | 0.6566 |
| P | KNN | 0.7778 | 0.7041 | 0.7113 | 0.6970 |
| KB | RF | 0.7893 | 0.7027 | 0.7558 | 0.6566 |
| VT | KNN | 0.7816 | 0.7016 | 0.7283 | 0.6768 |

## Conclusion

Le meilleur modele final est donc :

```text
Random Forest + Variance Threshold
```

Ce choix est le plus coherent avec l'objectif du projet, car il maximise le F1-score tout en conservant la meilleure accuracy test et le meilleur recall.
