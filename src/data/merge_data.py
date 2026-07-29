import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.dirname(__file__)
    )
)

from load_data import load_data

def merge_data():

    train, test, meal, center = load_data()

    # Merge train data with meal information
    train = train.merge(
        meal,
        on="meal_id",
        how="left"
    )

    # Merge train data with center information
    train = train.merge(
        center,
        on="center_id",
        how="left"
    )

    return train


if __name__ == "__main__":

    df = merge_data()

    print("Merged Data Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)