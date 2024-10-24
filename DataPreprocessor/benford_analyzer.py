import matplotlib
matplotlib.use('Agg')  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare
import base64
from io import BytesIO  

class BenfordAnalyzer:
 

    def __init__(self):

        pass
    
    def extract_first_digits(self, data: pd.Series) -> pd.Series:

        first_digits = data.astype(str).str.lstrip('-0.').str[0]
  
        first_digits = first_digits[first_digits.str.contains(r'\d')]
        return first_digits.astype(int)
    def calculate_benford_distribution(self) -> pd.Series:
     
        digits = np.arange(1, 10)
        benford_probs = np.log10(1 + 1 / digits)
        return pd.Series(benford_probs, index=digits)
    
    def calculate_empirical_distribution(self, first_digits: pd.Series) -> pd.Series:
  
        counts = first_digits.value_counts().sort_index()
        total = counts.sum()
        empirical_probs = counts / total
        return empirical_probs
    def calculate_empirical_distribution(self, first_digits: pd.Series) -> pd.Series:
  
        counts = first_digits.value_counts().sort_index()
        total = counts.sum()
        empirical_probs = counts / total
        return empirical_probs
    def chi_square_test(self, empirical_probs: pd.Series, benford_probs: pd.Series, total_count: int) -> dict:
   
        observed = empirical_probs * total_count
        expected = benford_probs * total_count
        chi_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        return {"chi_stat": chi_stat, "p_value": p_value}
    def analyze(self, data: pd.DataFrame, column: str) -> dict:
 
        if column not in data.columns:
            raise ValueError(f"Kolumna '{column}' nie istnieje w danych.")

     
        first_digits = self.extract_first_digits(data[column].dropna())

 
        benford_probs = self.calculate_benford_distribution()
        empirical_probs = self.calculate_empirical_distribution(first_digits)


        empirical_probs = empirical_probs.reindex(benford_probs.index, fill_value=0)


        test_results = self.chi_square_test(empirical_probs, benford_probs, len(first_digits))


        return {
            "empirical_probs": empirical_probs,
            "benford_probs": benford_probs,
            "test_results": test_results
        }
    
    def plot_distribution(self, empirical_probs: pd.Series, benford_probs: pd.Series, title: str = "Analiza Prawa Benforda") -> str:
    
    
        plt.figure(figsize=(10, 6))
        digits = np.arange(1, 10)
        width = 0.4

        plt.bar(digits - width/2, empirical_probs.values * 100, width=width, label='Rozkład empiryczny', align='center')
        plt.bar(digits + width/2, benford_probs.values * 100, width=width, label='Rozkład Benforda', align='center')

        plt.xticks(digits)
        plt.xlabel('Pierwsza cyfra')
        plt.ylabel('Częstość (%)')
        plt.title(title)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()


        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image_png = buf.getvalue()
        buf.close()

        graphic = base64.b64encode(image_png)
        graphic = graphic.decode('utf-8')

        return graphic