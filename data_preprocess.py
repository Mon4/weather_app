import pandas as pd
import json


# select and save columns from last_data
with open('data/last_data.json') as f:
    data = json.load(f)

keys = ['time', 'temp_c', 'wind_kph', 'wind_dir', 'pressure_mb', 'precip_mm']
last_df = pd.DataFrame(columns=keys)
row = []

for d in range(0, 3):
    for h in range(0, 24):
        for key in keys:
            row.append(data['forecast']['forecastday'][d]['hour'][h].get(key))
        last_df.loc[len(last_df)] = row
        row = []

last_df.to_csv('data/last_data_small.csv', index=False)


# select and save data from history_data
history_df = pd.read_excel('data/history_data.xlsx')
hisory_keys = ['time', 'temperature_2m (°C)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)', 'surface_pressure (hPa)',
        'precipitation (mm)']

history_df_small = history_df[hisory_keys]
history_df_small = history_df_small.dropna()

history_df_small.columns = keys

history_df_small.to_csv('data/history_data_small.csv', index=False)
