from sklearn.ensemble import GradientBoostingRegressor


def gradient_boosting():

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )

    return model