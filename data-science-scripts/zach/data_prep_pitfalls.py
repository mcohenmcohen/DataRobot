# WRONG
import pandas as pd
df = pd.DataFrame(
  {
    'ID': [1, 1, 1, 2, 2],
    'Date': ['2021-01-01', '2021-02-01', '2021-03-01', '2022-01-01', '2022-02-01'], 
    'Churn': [0, 0, 1, 0, 0],
  })
df['Date'] = pd.to_datetime(df['Date'])
df['Start'] = df['Date'].groupby(df['ID']).transform('min')
df['Tenure'] = df['Date'] - df['Start']

df_new = pd.DataFrame(
  {
    'ID': [2],
    'Date': ['2022-03-01'],
  })

df_new['Date'] = pd.to_datetime(df_new['Date'])
df_new['Start'] = df_new['Date'].groupby(df_new['ID']).transform('min')
df_new['Tenure'] = df_new['Date'] - df_new['Start']

# Correct
import pandas as pd
df = pd.DataFrame(
  {
    'ID': [1, 1, 1, 2, 2],
    'Date': ['2021-01-01', '2021-02-01', '2021-03-01', '2022-01-01', '2022-02-01'], 
    'Churn': [0, 0, 1, 0, 0],
  })
df['Date'] = pd.to_datetime(df['Date'])
df_start_dates = df.groupby(df['ID']).min()
df = df.merge(df_start_dates, on='ID')
df['Tenure'] = df['Date_x'] - df['Date_y']

df_new = pd.DataFrame(
  {
    'ID': [2],
    'Date': ['2022-03-01'],
  })
df_new['Date'] = pd.to_datetime(df_new['Date'])
df_new = df_new.merge(df_start_dates, on='ID')
df_new['Tenure'] = df_new['Date_x'] - df_new['Date_y']
