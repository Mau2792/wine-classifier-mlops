import os
from typing import  Any
import joblib

def save_model(model: Any, path: str) -> None:
    """
    Saves the model at the specified path
    """
    # Identifying the directory
    directory = os.path.dirname(path)

    # Creating the directory if necessary
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Serialzing and saving the model
    joblib.dump(model, path)

    print(f"Model saved at {path}")


def load_model(path: str) -> Any:
    """
    Loads the model stored at the path passed as argument
    """
    if not os.path.exists(path):
        raise FileNotFoundError
    
    return joblib.load(path)