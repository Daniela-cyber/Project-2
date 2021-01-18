import pandas


def prepare_weather_data():
    weather_data = pandas.read_csv("Weather data.csv", sep=",")
    # drop duplicates
    weather_data = weather_data.drop_duplicates(subset="dt")

    # remove last 2 hours since we have no associated energy data for it
    weather_data = weather_data[:-2]

    # interpret date column from string to UTC
    weather_data["dt_iso"] = pandas.to_datetime(weather_data["dt_iso"]).dt.tz_localize("UTC")

    # reset index
    weather_data = weather_data.reset_index()

    return weather_data


def prepare_energy_data():
    energy_data = pandas.read_csv("Energy_use_houshold_summary.csv", sep=",")

    # convert to utc
    energy_data["Date"] = pandas.to_datetime(energy_data["Date"]).dt.tz_localize("Europe/Vienna").dt.tz_convert("UTC")

    # remove first three entries since we have no associated weather data for it
    energy_data = energy_data[3:]

    # reset index
    energy_data = energy_data.reset_index()

    # sum over quarter hours
    kwh = energy_data.groupby(energy_data.index // 4).sum()["kWh"]

    return kwh


def prepare_data():
    # merge the two datasets
    data = prepare_weather_data()
    data["kWh"] = prepare_energy_data()

    return data
