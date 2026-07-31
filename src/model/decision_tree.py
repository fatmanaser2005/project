from sklearn.tree import DecisionTreeRegressor


def decision_tree():

    model = DecisionTreeRegressor(
        random_state=42
    )

    return model