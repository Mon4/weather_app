import pandas as pd

from select_history_weather import select_history_weather
from select_current_weather import select_current_weather


def main():
    # select_current_weather(48)
    # select_history_weather()

    dfh = pd.read_csv('data/history_data_small.csv')
    dfc = pd.read_csv('data/last_data.csv')


if __name__ == "__main__":
    main()
