from sklearn.datasets import load_wine
import pandas as pd
import os
from typing import Tuple
from sklearn.model_selection import train_test_split

def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from sklearn library and converts it
    in pandas DataFrame saving it at the speicific data path
    passed as argument
    """
    print("Loading raw data...")

    # Loading raw data
    wine_dataset = load_wine()
    data = wine_dataset.data

    print(f"TARGET NAMES: {wine_dataset.target_names}")
    # Saving feature names to name the columns of our DataFrame
    feature_names = wine_dataset.feature_names
    # Creating pandas DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    # Adding result column
    results = wine_dataset.target
    df['target'] = results

    # Creating the directory to save raw data if it doesn't exist already
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Converting df to csv and saving it at the directory we just created
    df.to_csv(data_path, index=False)

    print(f"Data successfully stored to {data_path} and converted to DataFrame")
    
    return df

def split_data(df: pd.DataFrame, test_size: float, random_state: int,  processed_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the DataFrame in training and test sets, with their respective target results.
    Returns X_train, X_test, y_train, y_test
    """
    print("Splitting the data...")
    
    # Splitting data (X) and results (y)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Further splitting in X_train, X_test, y_train, y_test using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    
    print("Training and test sets splitted")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {y_test.shape[0]}")

    # Saving training dataset in the apposite directory
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    X_train.to_csv(f"{processed_path}", index=False)


    return X_train, X_test, y_train, y_test
    
