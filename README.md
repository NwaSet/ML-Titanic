# ML-Titanic

## The Project :

Building a complete model for the Titanic dataset to predict whether a passenger will survive or not. Starting with data analysis, followed by data preprocessing and feature selection, and finally training various models with performance evaluation.

### The pipeline :

**how to use the pipeline :**

1. architecture
```
pipeline/
│
├── src/
│   ├── const.py
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── pipeline.py
│
└── outputs/
    ├── model_accuracy.csv
    ├── best_model.joblib
    │
    └── models/
        ├── DT_KB.joblib
        ├── DT_P.joblib
        ├── DT_VT.joblib
        ├── KNN_KB.joblib
        ├── KNN_P.joblib
        ├── KNN_VT.joblib
        ├── RF_KB.joblib
        ├── RF_P.joblib
        └── RF_VT.joblib
```

2. how to start a pipeline :

    1. change const.py file with the new csv PATH

    2. use the commande
    
    ```bash
    python pipeline/src/pipeline.py
    ```

## How To Install The Project :

1. clone git repository 
```bash
git clone https://github.com/NwaSet/AIProject
```

---

2. create a Virtual Environment 
```bash
python -m venv venv
```

---

activate venv  :
```bash
venv\Scripts\activate
```

---

3. install the requirements on your Virtual Environment
```bash
python -m pip install -r requirements.txt
```

## Author Of The Project :
- [NwaSet](https://github.com/NwaSet).
- [Nhanyo](https://github.com/Nhanyo).

## LICENSE :
This project is licensed under the MIT License.