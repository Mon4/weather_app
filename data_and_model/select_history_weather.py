import pandas as pd


# history_data
def select_history_weather():
    df = pd.read_excel('data/history_data.xlsx')

    history_keys = ['time', 'temperature_2m (°C)', 'wind_speed_10m (km/h)', 'wind_direction_10m (°)',
                    'surface_pressure (hPa)', 'precipitation (mm)', 'snowfall (cm)', 'relative_humidity_2m (%)',
                    'cloud_cover (%)', 'apparent_temperature (°C)', 'is_day ()']
    keys = ['time', 'temp_c', 'wind_kph', 'wind_degree', 'pressure_mb', 'precip_mm', 'snow_cm', 'humidity', 'cloud',
            'feelslike_c', 'is_day']

    df = df[history_keys]

    # prepare data before saving
    df.columns = keys

    # save data
    df.to_csv('data/history_data_small.csv', index=False)


