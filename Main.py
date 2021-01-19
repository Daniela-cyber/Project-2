from InputProcessing import prepare_data
from sklearn.model_selection import train_test_split
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def encode_cyclical_weekday(data):
    data["day_norm"] = 2 * pi * data["day"] / 7
    data["day_sin"] = np.sin(data["day_norm"])
    data["day_cos"] = np.cos(data["day_norm"])

def main():
    hyperparameters = {
        "contamination": 0.01
    }
    data = prepare_data()
    encode_cyclical_weekday(data)

    data = data[[
        "dt",
        "temp", "pressure", "humidity", "wind_speed", "wind_deg",
                 "rain_1h",
                 "snow_1h",
                 "clouds_all",
                 "kWh",
                 "precipitation",
                 "day_sin", "day_cos"]]

    plt.figure(figsize=(12, 10))
    cor = data.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

    train, test = train_test_split(data)



main()