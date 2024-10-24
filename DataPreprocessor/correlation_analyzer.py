from abc import ABC, abstractmethod
from typing import List

import numpy as np
from feature_engine.selection import SmartCorrelatedSelection

import pandas as pd

class CorrelationAnalyzer():

    def __init__(self, method: str = 'pearson', threshold: float = 0.8,
                 selection_method: str = "variance"):
        self.default_method = method
        self.default_threshold = threshold
        self.deafult_selection_method = selection_method
    
    def process(self, data: pd.DataFrame, threshold, method, selection_method) -> pd.DataFrame:
        scs = SmartCorrelatedSelection(threshold=threshold, method=method, selection_method=selection_method)
        return scs.fit_transform(data)

    def list_correlated_features(self,threshold,method) -> None:
        scs = SmartCorrelatedSelection(method=method,threshold=threshold)
        print(f"Skorelowane cechy na poziomie: {self.default_threshold}\n")
        for i, feature_set in enumerate(scs.correlated_feature_sets_, start=1):
            print(f"Zestaw {i}:")
            for feature in feature_set:
                print(f" - {feature}")
            print()

    def get_correlation_statistics(self, data: pd.DataFrame) -> None:
        corr_matrix = data.corr()
        corr_matrix_without_self = corr_matrix.copy()
        np.fill_diagonal(corr_matrix_without_self.values, np.nan)
        average_correlation = corr_matrix_without_self.mean().dropna()
        print("Średnia wartość korelacji dla każdej cechy:")
        print(average_correlation)
        max_correlation = corr_matrix_without_self.max().dropna().max()
        print(f"Maksymalna wartość korelacji między cechami (ignorując 1 na przekątnej): {max_correlation}")

    def columns_to_remove(self, data: pd.DataFrame,threshold,method,selection_method) -> list:
        scs = SmartCorrelatedSelection(threshold=threshold, method=method, selection_method=selection_method)
        scs.fit(data)
        return scs.features_to_drop_

    def validate_settings(self, method, threshold, selection_method) -> bool:
        M = self._is_valid_method(method)
        S = self._is_valid_selection_method(selection_method)
        T = self._is_valid_threshold(threshold)
        return M and S and T
    
    def _is_valid_method(self,method:str) -> bool:
        return method in ['pearson','spearman','kendall']
    def _is_valid_selection_method(self,selection_method:str) -> bool:
        return selection_method in ['variance','missing_values','cardinality']
    def _is_valid_threshold(self,threshold:float) -> bool:
        return threshold >= 0 and threshold <= 1