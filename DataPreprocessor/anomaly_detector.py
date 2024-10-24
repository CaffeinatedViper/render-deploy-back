import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class AnomalyDetector:


    def __init__(self, contamination=0.05, random_state=42):

        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.imputer = None

    def fit(self, data: pd.DataFrame):

    
        self.imputer = SimpleImputer(strategy='mean')
        data_imputed = self.imputer.fit_transform(data)

    
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data_imputed)


        self.model = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        self.model.fit(data_scaled)

    def predict(self, data: pd.DataFrame) -> pd.Series:
 

        data_imputed = self.imputer.transform(data)

        data_scaled = self.scaler.transform(data_imputed)
        predictions = self.model.predict(data_scaled)
        return pd.Series(predictions, index=data.index)

    def get_anomaly_scores(self, data: pd.DataFrame) -> pd.Series:
 
        data_imputed = self.imputer.transform(data)

        data_scaled = self.scaler.transform(data_imputed)
        scores = self.model.decision_function(data_scaled)
        return pd.Series(scores, index=data.index)
