# Libs
import pandas as pd

# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def build_models(modelo: str):
    '''
    Docstring for build_models
    Construtor de modelos.
    Insira o modelo desejado:
    - decisiontree = DecicisonTreeClassifier()
    - randomforest = RandomForestClassifier()
    - xgboost = XGBClassifier()
    - lgmboost = LGBMClassifier()
    :param model: Description
    '''

    if modelo.lower() == 'decisiontree':
        model = DecisionTreeClassifier()
    elif modelo.lower() == 'randomforest':
        model = RandomForestClassifier()
    elif modelo.lower() == 'xgboost':
        model = XGBClassifier()
    elif modelo.lower() == 'lgmboost':
        model = LGBMClassifier()

    return model