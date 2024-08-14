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

## 2. Preprocessing

Replacing nulls with the previous and following value, casting date and time column to datetime.

## 2.5. Tests

### 1. Stationary test - Augmented Dickey-Fuller (ADF)

From this test we got to know that almost all columns are stationary. Only is_day is non-stationary. 

### 2. Autocorrelation test - ACF and PCAF

#### Seasonoality:
feelslike_c, humidity, is_day, temp_c

![alt text](ACF_temp_c_48.png)
![alt text](ACF_is_day_48.png)


#### Decay in Correlation: 

cloud, precip_mm, pressure_mb, snow_cm, wind_degree, wind_kph

![alt text](ACF_cloud_48.png)

![alt text](ACF_precip_mm_48.png)

## 3. Model

## 4. Website
