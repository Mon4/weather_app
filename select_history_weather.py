import pandas as pd


# history_data
def select_history_weather():
    df = pd.read_excel('data/history_data.xlsx')

    history_keys = ['time', 'temperature_2m (°C)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)',
                    'surface_pressure (hPa)', 'precipitation (mm)']
    keys = ['time', 'temp_c', 'wind_kph', 'wind_dir', 'pressure_mb', 'precip_mm']

    df = df[history_keys]

    # rename columns
    df.columns = keys

    df.to_csv('data/history_data_small.csv', index=False)


