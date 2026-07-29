import pandas as pd
import os


def load_data():

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )

    data_path = os.path.join(BASE_DIR, "data", "raw")

    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    test = pd.read_csv(os.path.join(data_path, "test.csv"))
    meal = pd.read_csv(os.path.join(data_path, "meal_info.csv"))
    center = pd.read_csv(os.path.join(data_path, "fulfilment_center_info.csv"))

    return train, test, meal, center


if __name__ == "__main__":

    train, test, meal, center = load_data()

    print("Train:", train.shape)
    print("Test:", test.shape)
    print("Meal:", meal.shape)
    print("Center:", center.shape)