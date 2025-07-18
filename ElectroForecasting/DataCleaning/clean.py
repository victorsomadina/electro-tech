import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

data = pd.read_csv(r'C:\Users\hp\Downloads\redertelecomchurn\ElectroForecasting\ElectroTech Forecasting Data.csv')

print(data.head())
print(data['Category'].unique())
print(f"{data.isnull().sum()}\n")
print(f"{data.duplicated().sum()}\n")
print(f"{data.dtypes}\n")

data['Date'] = data['Date'].astype("datetime64[ns]")
print(data['Date'].dtype)

outlier_columns = data.select_dtypes(exclude=['object', 'datetime64[ns]']).iloc[0:, 1:].columns.to_list()

for column in outlier_columns:
    plt.boxplot(data[column])
    plt.title(column)
    plt.show()