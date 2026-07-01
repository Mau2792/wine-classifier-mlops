from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from typing import Any

def train_model(df: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int, max_depth: int, random_state: int) -> Any:
    """
    Initializes and train Random Forest CLassifier
    ---------------------------------------------------
    Args:
    - df, DataFrame
    - X_train, training dataset
    - y_train, training targets
    - n_estimators, number  of trees in the forest,
    - max_depth, depth of a single tree
    - random_state, controls the randomness of the bootstrapping and features selection
    ---------------------------------------------------
    Returns the trained model
    """

    # Initializing the model
    print("Initializing the model...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    # Training the model on X_train and y_train
    print("Traininig the model...")
    model.fit(X_train, y_train)

    return model
