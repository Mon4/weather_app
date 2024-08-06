# Krakow_weahter_app
## 1. Data

history data: hourly data from 1.01.2019 to 11.07.2024 from Berlin(Germany)

https://open-meteo.com/en/docs#hourly=temperature_2m&timezone=Europe%2FBerlin

API from Cracow two days back from current date. In ending we need 48 h time series.

https://www.weatherapi.com/

API Key:
ee76052c13e141698b2112304242207


Interresting variables:

| Names in historic data  |     Names in API (last) data     |
| :---------------------: |:--------------------------------:|
| temp_c                  |       temperature_2m (°C)        |
| wind_kph                |      wind_speed_10m (km/h)       |
| wind_dir                |      wind_direction_10m (°)      |
| pressure_mb (hPA)       |      surface_pressure (hPa)      |
| precip_mm               |        precipitation (mm)        |
| snow_cm                 |             snowfall             |
| humidity                |   relative_humidity_2m (%)       |
| cloud                   |         cloud_cover (%)          |
| feels_like_c            |    apparent_temperature (°C)     |
| is_day                  |             is_day()             |

and maybe more...

## 2. Preprocessing

## 3. Model

## 4. Website
