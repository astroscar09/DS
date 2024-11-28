import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

class ExploratoryAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        if self.file_path.endswith('.csv'):
            return pd.read_csv(self.file_path)
        elif self.file_path.endswith('.txt'):
            return pd.read_csv(self.file_path, delimiter='\s%')
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .txt file.")

    def show_null_values(self):
        null_values = self.data.isnull().sum()
        print("Null values in each column:\n", null_values)

    def show_column_names(self):
        print("Column names:\n", self.data.columns.tolist())

    def show_descriptive_statistics(self):
        print("Descriptive statistics:\n", self.data.describe())

    def plot_histogram(self, column_name):
        if column_name in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[column_name].dropna(), kde=True)
            plt.title(f'Histogram of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.show()
        else:
            print(f"Column '{column_name}' not found in the data.")

    def plot_correlation_matrix(self):
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()

# Example usage:
# analysis = ExploratoryAnalysis('/path/to/your/file.csv')
# analysis.show_null_values()
# analysis.show_column_names()
# analysis.show_descriptive_statistics()
# analysis.plot_histogram('column_name')
# analysis.plot_correlation_matrix()