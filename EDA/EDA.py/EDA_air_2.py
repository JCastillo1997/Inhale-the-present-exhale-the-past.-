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
df=pd.read_csv('C:/Users/Usuario/Documents/Inhale-the-present-exhale-the-past.-/Data/Unclean_data/Unclean_data/aire_madrid.csv')
df=df.dropna()
# We separate the first and second columns based only on when ZBEDEP and ZBE where created.
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

df=df.drop('Unnamed: 0', axis=1)
df = df[~df[columns_to_check].apply(lambda row: row.str.contains('n')).any(axis=1)]
for col in columns_to_modify2:
    df[col] = df[col].str.replace(',', '.').astype(float)
df = df.drop(df.columns[df.columns.str.startswith('v')], axis=1)
df.head()
After studying the DataFrames we detelte the 'V' columns
cross_1= pd.crosstab(df['municipio'],df['magnitud'])
cross_1
df=df.drop('Unnamed: 0', axis=1)
df = df[~df[columns_to_check].apply(lambda row: row.str.contains('n')).any(axis=1)]

for col in columns_to_modify2:
    df[col] = df[col].str.replace(',', '.').astype(float)
df[columns_to_modify2] = scaler.fit_transform(df[columns_to_modify2])
df=df.drop(columns_to_check,axis=1)
# df.to_csv('aire_global',index=False)
# df_antes.to_csv('aire_antes',index=False)
# df_despues.to_csv('aire_después',index=False)
df.rename(columns={'provincia':'province','municipio':'municipality','estacion':'station','estacion':'station','magnitud':'magnitude',
                   'punto_muestreo':'measuring_point','ano':'year','mes':'month','dia':'day'}, inplace=True)
df_id=df[['province','municipality','magnitude','month']]
df_id=df_id.rename(columns={'province':'province_id','municipality':'municipality_id','magnitude':'magnitude_id','month':'month_id'})
df.head()
df_id
municipality_map= {16:'el_atazar',47:'collado_villalba',6:'alcobendas',58:'fuenlabrada',171:'villa_del_prado',
                   5:'alcala_de_henares',92:'mostoles',102:'orusco_de_tajuña',45:'colmenar_viejo',148:'torrejon_de_ardoz',
                   74:'leganes',80:'majadahonda',13:'aranjuez',123:'rivas_vaciamadrid',9:'algete',49:'coslada',
                   180:'villarejo_de_salvanes',14:'arganda_del_rey',161:'valdemoro',67:'guadalix_de_la_sierra',
                   133:'san_martin_de_valdeiglesias',7:'alcorcon',65:'getafe',120:'puerto_de_cotos',115:'pozuelo_de_alarcon',
                   134:'san_sebastian_de_los_reyes',127:'las_rozas',106:'parla'}
df.head()

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
df['province'] = df['province'].map(province_map)
df['municipality'] = df['municipality'].map(municipality_map)
magnitude_map={14:'ozone',7:'nitrogen_moxide',8:'nitrogen_dioxide',10:'particles_in_suspension(<_PM10)',
              12:'nitrogen_oxides',9:'particles_in_suspension(<_PM2,5)',6:'carbon_monoxide',1:'sulfur_dioxide',
              30:'benzene',20:'toluene',44:'non_methane_hydrocarbons',42:'total_hydrocarbons',
              431:'metaparaxylene'}
df['magnitude'] = df['magnitude'].map(magnitude_map)

split_values = df['measuring_point'].str.split('_')
extracted_numbers = split_values.str[2]
df['tech'] = extracted_numbers
df['measuring_point'] = df['measuring_point'].str.replace(r'_(\d+)$', '', regex=True)

df['measuring_point'] = df['measuring_point'].str.replace(r'_(\d+)$', '', regex=True)
technical_description_map={'8':'quimioluminescence','49':'beta_absortion','6':'uv_absortion',
                           '59':'gas_cromatography','2':'flame_ionization','48':'non_dispersed_infrared_espectometry',
                           '38':'uv_fluorescence'}
df['tech'] = df['tech'].map(technical_description_map)

df_1 = df[['year', 'month', 'day', 'tech','tech_id', 'h01', 'h02', 'h03', 'h04', 'h05', 'h06', 'h07', 'h08', 'h09', 'h10',
            'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24']]
df_1.to_csv('erd_1.csv',index=False)

merged_df = df.merge(df_id, left_index=True, right_index=True, how='inner')
df_sql_1= merged_df[['year', 'month', 'day','magnitude','tech', 'h01', 'h02', 'h03', 'h04', 'h05', 'h06', 'h07', 'h08', 'h09', 'h10',
            'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24','province_id']]
# df_sql_1.to_csv('df_sql_1.csv',index=False)
df_sql_2= merged_df[['province']]

df_sql_2=df_sql_2.groupby('province').mean().reset_index()
df_sql_2.head()
df_sql_2.reset_index(drop=False, inplace=True)
df_sql_2.rename(columns={'index':'province_id'}, inplace=True)
df_sql_2['province_id'] += 1

df_sql_2.reset_index(inplace=True)
df_sql_2.rename(columns={'index':'province_id'}, inplace=True)
df_sql_2.head()
df_sql_2.to_csv('df_sql_2.csv',index=False)
# df_sql_2.to_csv('df_sql_2.csv',index=False)
# df_sql_2.to_csv('df_sql_2.csv',index=False)
province_map2 = {28 : 1}
merged_df['province_id'] = merged_df['province_id'].map(province_map2)

municipality_map2= {16:1,47:2,6:3,58:4,171:5,
                   5:6,92:7,102:8,45:9,148:10,
                   74:11,80:12,13:14,123:15,9:16,49:17,
                   180:18,14:19,161:20,67:21,
                   133:22,7:23,65:24,120:25,115:26,
                   134:27,127:28,106:29}
merged_df['municipality_id'] = merged_df['municipality_id'].map(municipality_map2)

df_sql_3=merged_df[['municipality','province_id']]
# df_sql_3.to_csv('df_sql_3.csv',index=False)
df_sql_4=merged_df[['station','municipality_id']]
#d f_sql_4.to_csv('df_sql_4.csv',index=False)
province_map2 = {28 : 1}
df_sql_1['province_id'] = df_sql_1['province_id'].map(province_map2)
df_sql_1.reset_index(inplace=True)
df_sql_1.rename(columns={'index':'pollution_id'}, inplace=True)
#df_sql_1.to_csv('df_sql_1.csv',index=False)
#merged_df.to_csv('pollution_data.csv',index=False)


merged_df['municipality'] = merged_df['municipality'].str.replace('_', ' ')

# Display the updated dataframe
merged_df.municipality.unique()


# merged_df.to_csv('pollution_madrid.csv',index=False)
d
df_12=merged_df[merged_df['municipality']==str('pozuelo de alarcon')]
merged_df.magnitude.unique()