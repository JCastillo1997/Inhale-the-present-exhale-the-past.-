import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats.contingency import association
from scipy.stats import spearmanr, pearsonr
pd.set_option('display.max_columns', None)
from scipy.stats import f_oneway, norm, f, chi2, chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
scaler = MinMaxScaler()
df_cars=pd.read_csv('C:/Users/Usuario/Documents/Inhale-the-present-exhale-the-past.-/Data/Clean_data/Registered_car_sales.csv', encoding= 'utf-8')
df_cars.head()
df_kars=df_cars['Gasolina']+df_cars['Diésel']
df_kars2=df_cars['Gasolina']+df_cars['Diésel']+df_cars['Híbridos, eléctricos y otros']
df_kars = df_kars.to_frame()
df_kars2 = df_kars2.to_frame()
merged_df= df_kars.join(df_kars2, lsuffix='_kars', rsuffix='_kars2')
df_cars.columns
df_cars=df_cars.join(merged_df,lsuffix='df_cars')
df_cars = df_cars.rename(columns={'0_kars':'Total_combustion','0_kars2':'Total_Sold'})
df_cars.dropna()

df_cars = df_cars.rename(columns={'0_kars':'Total_combustion','0_kars2':'Total_Sold'})
df_kars= df_cars.groupby('Año').agg({'Total_Sold':'value_counts'})
df_kars
result = df_cars.groupby('Año')['Total_Sold'].sum().reset_index()
df_kars= df_cars.groupby('Año').agg({'Total_Sold':'value_counts'})

type(result)
df_kars=df_kars[df_kars['Total_Sold']==type(str)]
df_kars
df_cars['Año'] = df_cars['Año'].astype(str)
df_cars = df_cars[~df_cars['Año'].str.contains('a')]
df_cars = df_cars[~df_cars['Año'].str.contains('u')]
df_cars = df_cars[~(df_cars['Total_Sold'] == 969.500)]
df_cars.Año.unique()
df_cars = df_cars[~(df_cars['Total_Sold'] == 969.500)]
sns.barplot(x='Año', y='Total_Sold', data=df_cars)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df_cars.rename(columns={'Año':'year','Periodo':'month','Híbridos, eléctricos y otros':'eco','Gasolina':'gasoline', 
                'Diésel':'diesel','Total_combustion':'total_combustion','Total_Sold':'total_sold'},inplace=True)
month_map = {
    'Enero' : 'january',
    'Febrero': 'february',
    'Marzo': 'march',
    'Abril': 'april',
    'Mayo': 'may',
    'Junio': 'june',
    'Julio': 'july',
    'Agosto': 'august',
    'Septiembre': 'september',
    'Octubre': 'october',
    'Noviembre': 'november',
    'Diciembre': 'december'
}


df_cars['month'] = df_cars['month'].map(month_map)
df_cars.to_csv('cars_sales.csv',index=False)