import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plp
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

df_cars=pd.read_csv('C:/Users/Usuario/Documents/Inhale-the-present-exhale-the-past.-/Data/Clean_data/Matriculaciones.csv', encoding= 'utf-8',sep=';')
df_budget_mad=pd.read_csv('C:/Users/Usuario/Documents/Inhale-the-present-exhale-the-past.-/Data\Clean_data/Presupuestos_madrid.csv', encoding='utf-8')
df_budget_sp=pd.read_csv('C:/Users/Usuario/Documents/Inhale-the-present-exhale-the-past.-/Data/Clean_data/Presupuestos.csv', encoding='latin1',sep=';')

df_budget_mad.head()
df_budget_mad.rename(columns={'Año':'year','Id Política':'policy','Nombre Política':'policy_name','Id Programa':'program_id',
                          'Nombre Programa':'program_name','Presupuesto Gastos':'spending_budget','Gastos Reales':'total_spent'},
                            inplace=True)
df_budget_sp.dropna(inplace=True)
df_budget_sp = df_budget_sp.applymap(lambda x: x.lower() if type(x) == str else x)
df_budget_sp.head()
df_budget_sp.rename(columns={'ESTADO DE EJECUCIÓN DEL PRESUPUESTO DE GASTOS':'budget_center',
                             'Unnamed: 1':'center_desc','Unnamed: 2':'sector','Unnamed: 3':'sector_desc',
                             'Unnamed: 4':'managment_center','Unnamed: 5':'managment_center_desc','Unnamed: 6':'program',
                             'Unnamed: 7':'program_desc','Unnamed: 8':'chapter','Unnamed: 9 ':'chapter_desc',
                             'Unnamed: 10':'initial_credit','Unnamed: 11':'actual_credit','Unnamed: 12':'credit_mod',
                             'Unnamed: 13':'approved_budget','Unnamed: 14':'balance','Unnamed: 15':'allocated_budget',
                             'Unnamed: 16':'authorized_budget','Unnamed: 17':'obligations','Unnamed: 18':'operable_budget'},
                             inplace=True)
df_budget_sp.head()
df_budget_sp= df_budget_sp.drop(df_budget_sp.index[0])
df_budget_sp.head()

df_budget_sp.reset_index(drop=True, inplace=True)



df_budget_sp.to_csv('df_budget_sp.csv',index=False)
df_budget_mad.to_csv('df_budget_mad.csv',index=False)