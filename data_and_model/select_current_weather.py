from __future__ import print_function
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint
from datetime import datetime, timedelta
import pandas as pd


def select_columns(data: dict) -> pd.DataFrame:
    keys = ['time', 'temp_c', 'wind_kph', 'wind_dir', 'pressure_mb', 'precip_mm']
    last_df = pd.DataFrame(columns=keys)
    row = []

    # select key columns
    for d in range(0, 3):
        for h in range(0, 24):
            for key in keys:
                row.append(data['forecast']['forecastday'][d]['hour'][h].get(key))
            last_df.loc[len(last_df)] = row
            row = []

    return last_df

# leave only defined amount of rows
def remove_rows(today: datetime, df: pd.DataFrame, hours: int = 48) -> pd.DataFrame:
    df['time'] = pd.to_datetime(df['time'])
    # days before today and till current hour
    df = df[(df['time'].dt.day != today.day) | (df['time'].dt.hour <= today.hour)]
    # defined amount of hours back
    df = df.iloc[-hours:].reset_index(drop=True)
    return df


def select_current_weather(hours: int) -> None:
    today = datetime.now()
    start_date = today - timedelta(hours=hours)

    # configure API key authorization: ApiKeyAuth
    configuration = swagger_client.Configuration()
    configuration.api_key['key'] = '945500161a894b3f93c81230242307'

    # create an instance of the API class
    api_instance = swagger_client.APIsApi(swagger_client.ApiClient(configuration))
    q = 'Cracow'  # str | Pass US Zipcode, UK Postcode, Canada Postalcode, IP address, Latitude/Longitude or city name
    dt = start_date
    end_dt = today
    lang = 'en'

    try:
        api_response = api_instance.history_weather(q, dt, end_dt=end_dt, lang=lang)
        pprint(api_response)

        df = select_columns(api_response)
        df = remove_rows(today, df)
        df.to_csv('data/last_data.csv', index=False)

    except ApiException as e:
        print("Exception when calling API->history_weather: %s\n" % e)
