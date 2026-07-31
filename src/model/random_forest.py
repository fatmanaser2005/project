from sklearn.ensemble import RandomForestRegressor


def random_forest():

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    return model