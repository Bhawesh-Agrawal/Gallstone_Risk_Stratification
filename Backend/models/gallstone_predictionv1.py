import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import joblib
import json
import os

class RandomForest:
    def __init__(self, data : dict):
        self.Age : int = data["Age"]
        self.Gender : bool = data['Gender']
        self.Comorbidity : int = data['Comorbidity']
        self.Coronary_Artery_Disease : bool = data["Coronary_Artery_Disease"]
        self.Hypothyroidism : bool = data["Hypothyroidism"]
        self.Hyperlipidemia : bool = data["Hyperlipidemia"]
        self.Diabetes : bool = data["Diabetes"]
        self.Height : int = data["Height"]
        self.Weight : float = data["Weight"]
        self.TBW :float = data["TBW"]
        self.ECW : float = data["ECW"]
        self.ICW : float = data["ICW"]
        self.ECF_TBW : int = data["ECF_TBW"]
        self.TBFR : float = data["TBFR"]
        self.LeanMass : float = data["LeanMass"]
        self.Protein : float = data["Protein"]
        self.VFR : int = data["VFR"]
        self.BoneMass : float = data["BoneMass"]
        self.MuscleMass : float = data["MuscleMass"]
        self.Obesity : float = data["Obesity"]
        self.TFC : float = data["TFC"]
        self.VFA : float = data["VFA"]
        self.VMA : float = data["VMA"]
        self.HFA : int = data["HFA"]
        self.Glucose : int = data["Glucose"]
        self.Total_Chloestrol : int = data["Total_Chloestrol"]
        self.LDL : int = data["LDL"]
        self.HDL : int = data["HDL"]
        self.Triglyceride : int = data["Triglyceride"]
        self.AST : int = data["AST"]
        self.ALT : int = data["ALT"]
        self.ALP : int = data["ALP"]
        self.creatinine : float = data["creatinine"]
        self.GFR : float = data["GFR"]
        self.C_reactive : float = data["C_reactive"]
        self.Hemoglobin : float = data["hemoglobin"]
        self.VitaminD : float = data["VitaminD"]