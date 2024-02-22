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
df=pd.read_csv('C:/Users/Usuario/Documents/Inhale-the-present-exhale-the-past.-/Data/Clean_data/aire_global_madrid.csv')
df_antes = pd.read_csv('C:/Users/Usuario/Documents/Inhale-the-present-exhale-the-past.-/Data/Clean_data/aire_antes.csv')
df_despues = pd.read_csv('C:/Users/Usuario/Documents/Inhale-the-present-exhale-the-past.-/Data/Clean_data/aire_después.csv')
df=df.dropna()
# We separate the first and second columns based only on when ZBEDEP and ZBE where created.
df_antes=df[df['ano']<2021]
df_despues= df[df['ano'] >= 2021]
df.head()
columns_to_check = ['v01', 'v02', 'v03', 'v04', 'v05', 'v06', 'v07', 'v08', 'v09', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24']
columns_to_modify2 = ['h01', 'h02', 'h03', 'h04', 'h05', 'h06', 'h07', 'h08', 'h09', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24']
# df_antes = df_antes[~df_antes[columns_to_check].apply(lambda row: row.str.contains('n')).any(axis=1)]
# df_despues = df_despues[~df_despues[columns_to_check].apply(lambda row: row.str.contains('n')).any(axis=1)]

"""
for col in columns_to_modify2:
    df_antes[col] = df_antes[col].str.replace(',', '.').astype(float)
      df_antes[col] = df_antes[col].str.replace(',', '0').astype(float)
"""
# now we work the second table 
"""
df_despues=df.drop('Unnamed: 0', axis=1)
df_despues = df_despues[~df_despues[columns_to_check].apply(lambda row: row.str.contains('n')).any(axis=1)]
for col in columns_to_modify2:
    df_despues[col] = df_despues[col].str.replace(',', '.').astype(float)
"""
df_despues[columns_to_modify2] = scaler.fit_transform(df_despues[columns_to_modify2])
df_antes[columns_to_modify2] = scaler.fit_transform(df_antes[columns_to_modify2])



After studying the DataFrames we detelte the 'V' columns
#df_despues=df_despues.drop(columns_to_check,axis=1)
#df_antes=df_antes.drop(columns_to_check,axis=1)

#H0 = Madrid rio ha funcionado, reduciendo los limites de C02 de los ultimos años
#H1 = Madrid rio no ha funcionado, los limites de C02 no se bhan reducido como se esperaba
cross_1= pd.crosstab(df['municipio'],df['magnitud'])
cross_1
estadistico, p_valor, ex, ddof = chi2_contingency(cross_1)

estadistico, p_valor
datos_ant=df_antes['magnitud']
datos_des=df_despues['magnitud']
df=df.drop('Unnamed: 0', axis=1)
df = df[~df[columns_to_check].apply(lambda row: row.str.contains('n')).any(axis=1)]

for col in columns_to_modify2:
    df[col] = df[col].str.replace(',', '.').astype(float)
df[columns_to_modify2] = scaler.fit_transform(df[columns_to_modify2])
df=df.drop(columns_to_check,axis=1)
# df.to_csv('aire_global',index=False)
# df_antes.to_csv('aire_antes',index=False)
# df_despues.to_csv('aire_después',index=False)

correlation, p_value = spearmanr(df["estacion"], df["magnitud"])
print("Coeficiente de correlación de Spearman:", correlation)
print("Valor p:", p_value)
correlation, p_value = pearsonr(df["estacion"], df["magnitud"])
print("Coeficiente de correlación de Pearson:", correlation)
print("Valor p:", p_value)
spearman= df_despues['punto_muestreo'].corr(df_despues['magnitud'], method='spearman')
pearson =df_antes['punto_muestreo'].corr(df_antes['magnitud'], method='pearson')
print(spearman)
print(pearson)
f_score, p_valor = f_oneway(df_antes['magnitud'], df_despues['magnitud'])
f_score,p_valor
print(datos_ant.isnull().sum())
print(datos_des.isnull().sum())

random_sample = df_antes.sample(n=154179)

datos_ant=random_sample['magnitud']
datos_des=df_despues['magnitud']
if df_despues.shape == random_sample.shape:
    print('yes')
else:
    print('no')
random_sample.head()
random_sample.head()
datos_des.head()
ttest_rel(df_despues['magnitud'],random_sample['magnitud'],alternative='less')

ttest_rel(df_despues['magnitud'],random_sample['magnitud'],alternative='less')
df.head()
df.rename(columns={'provincia':'province','municipio':'municipality','estacion':'station','estacion':'station','magnitud':'magnitude',
                   'punto_muestreo':'measuring_point','ano':'year','mes':'month','dia':'day'}, inplace=True)
df_id=df[['province','municipality','magnitude','month']]
df_id=df_id.rename(columns={'province':'province_id','municipality':'municipality_id','magnitude':'magnitude_id','month':'month_id'})
df_antes.rename(columns={'provincia':'province','municipio':'municipality','estacion':'station','estacion':'station','magnitud':'magnitude',
                   'punto_muestreo':'measuring_point','ano':'year','mes':'month','dia':'day'}, inplace=True)

municipality_map= {16:'el_atazar',47:'collado_villalba',6:'alcobendas',58:'fuenlabrada',171:'villa_del_prado',
                   5:'alcala_de_henares',92:'mostoles',102:'orusco_de_tajuña',45:'colmenar_viejo',148:'torrejon_de_ardoz',
                   74:'leganes',80:'majadahonda',13:'aranjuez',123:'rivas_vaciamadrid',9:'algete',49:'coslada',
                   180:'villarejo_de_salvanes',14:'arganda_del_rey',161:'valdemoro',67:'guadalix_de_la_sierra',
                   133:'san_martin_de_valdeiglesias',7:'alcorcon',65:'getafe',120:'puerto_de_cotos',115:'pozuelo_de_alarcon',
                   134:'san_sebastian_de_los_reyes',127:'las_rozas',106:'parla'}
province_map = {28:'madrid'}
month_map = {
    1: 'january',
    2: 'february',
    3: 'march',
    4: 'april',
    5: 'may',
    6: 'june',
    7: 'july',
    8: 'august',
    9: 'september',
    10: 'october',
    11: 'november',
    12: 'december'
}


df['month'] = df['month'].map(month_map)
df_antes['month'] = df_despues['month'].map(month_map)
df_antes['month'] = df_despues['month'].map(month_map)
df['province'] = df['province'].map(province_map)
df_antes['province'] = df_despues['province'].map(province_map)
df_antes['province'] = df_despues['province'].map(province_map)
df['municipality'] = df['municipality'].map(municipality_map)
df_antes['municipality'] = df_despues['municipality'].map(municipality_map)
df_antes['municipality'] = df_despues['municipality'].map(municipality_map)
magnitude_map={14:'ozone',7:'nitrogen_moxide',8:'nitrogen_dioxide',10:'particles_in_suspension(<_PM10)',
              12:'nitrogen_oxides',9:'particles_in_suspension(<_PM2,5)',6:'carbon_monoxide',1:'sulfur_dioxide',
              30:'benzene',20:'toluene',44:'non_methane_hydrocarbons',42:'total_hydrocarbons',
              431:'metaparaxylene'}
df['magnitude'] = df['magnitude'].map(magnitude_map)
df_antes['magnitude'] = df_despues['magnitude'].map(magnitude_map)
df_antes['magnitude'] = df_despues['magnitude'].map(magnitude_map)

split_values = df['measuring_point'].str.split('_')
extracted_numbers = split_values.str[2]
df['tech'] = extracted_numbers
df['measuring_point'] = df['measuring_point'].str.replace(r'_(\d+)$', '', regex=True)

df['measuring_point'] = df['measuring_point'].str.replace(r'_(\d+)$', '', regex=True)
technical_description_map={'8':'quimioluminescence','49':'beta_absortion','6':'uv_absortion',
                           '59':'gas_cromatography','2':'flame_ionization','48':'non_dispersed_infrared_espectometry',
                           '38':'uv_fluorescence'}
df['tech'] = df['tech'].map(technical_description_map)
df_antes['tech'] = df_despues['tech'].map(technical_description_map)
df_antes['tech'] = df_despues['tech'].map(technical_description_map)
df_1 = df[['year', 'month', 'day', 'tech', 'h01', 'h02', 'h03', 'h04', 'h05', 'h06', 'h07', 'h08', 'h09', 'h10',
            'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24']]
df_1.info()
df_1.to_csv('erd_1.csv',index=False)

merged_df = df.merge(df_id, left_index=True, right_index=True, how='inner')
merged_df.head()
df_2 =merged_df[['province',]]
df_3=merged_df[['municipality']]
dfsql2 = df_3.groupby("municipality").count()
dfsql2
dfsql2["municipality_id"] = [i for i in range(1, len(dfsql2)+1)]
dfsql2.drop(columns=['muniID'],inplace=True)
dfsql2=dfsql2.reset_index()
dfsql2.to_csv
df_1