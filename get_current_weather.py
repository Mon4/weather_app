from __future__ import print_function
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint
from datetime import date
import datetime
import json


def get_current_data(start_date, end_date): #yyyy-MM-dd
    # Configure API key authorization: ApiKeyAuth
    configuration = swagger_client.Configuration()
    configuration.api_key['key'] = 'ee76052c13e141698b2112304242207'

    # create an instance of the API class
    api_instance = swagger_client.APIsApi(swagger_client.ApiClient(configuration))
    q = 'Cracow' # str | Pass US Zipcode, UK Postcode, Canada Postalcode, IP address, Latitude/Longitude (decimal degree) or city name.
    dt = start_date
    end_dt = end_date
    lang = 'en'

    try:
        api_response = api_instance.history_weather(q, dt, end_dt=end_dt, lang=lang)
        pprint(api_response)
        # saving example of prediction data
        with open('data/last_data.json', 'w') as f:
            json.dump(api_response, f)
    except ApiException as e:
        print("Exception when calling APIsApi->history_weather: %s\n" % e)


def get_hourly_weather(hours: int, start_date: datetime, end_date: datetime):
    ...


today = date.today()
start_date = today - datetime.timedelta(days=2)
get_current_data(start_date, today)