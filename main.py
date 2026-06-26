from src.config_loader import load_config
from src.data_preprocessing import load_data, split_data
import pandas as pd

def main():
    print("\n"+ "="*45)
    print("BEGINNING OF THE PIPELINE")
    print("="*45)

    # Loading configuration (with no arguments passed it uses "config.yaml")
    config = load_config()

    # PREPROCESSING: Initializing DataFrame
    df = load_data(config['raw_data_path'])
    # Splitting data in train and test sets
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
