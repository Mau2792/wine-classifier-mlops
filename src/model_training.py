from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from typing import Any

def train_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> Any:
    """
    Initializes and train Random Forest CLassifier using GridSearchCV to determine best parameters
    ---------------------------------------------------
    Args:
    - df, DataFrame
    - X_train, training dataset
    - y_train, training targets
    - random_state, controls the randomness of the bootstrapping and features selection
    ---------------------------------------------------
    Returns the trained model
    """

    # Initializing the model
    print("Initializing the model...")

    base_model = RandomForestClassifier(
        random_state=random_state
    )

    # Selecting grid paramters to pass to GridSearchCV
    # 3 options for number of trees and 4 options for mac depth -> total of 12 different forest setting
    # max_depth=None means unlimited depth
    grid_parameters = {
        'n_estimators':[50, 100, 200],
        'max_depth':[3, 5, 10, None]
    }

    # Training and verifying each forest on training dataset X_train
    # cv (Cross Validation) is the number of subgroups in which we split our training dataset, to limit overfitting
    # It is then a multiplier of the total combination of options we defined in grid_parameters.
    # The total numbers of Training is now 3*4*5 [n_estimators size * max_depth size * cv]
    # n_jobs determines the number of CPU cores to use. n_jobs=-1 means 'use every core'
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=grid_parameters,
        cv=5,
        n_jobs=-1,
        verbose=1 # Print the progress to the terminal
    )

    print("Traininig the model...")
    grid_search.fit(X_train, y_train)

    print("\n" + "="*40)
    print("BEST PARAMETERS FOUND:")
    print(f"{grid_search.best_params_}")
    print("="*40 + "\n")
    

    return grid_search.best_estimator_
