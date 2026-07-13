from src.config_loader import load_config
from src.data_preprocessing import load_data, split_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.model_io import save_model
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
        random_state=config['random_state'],
        processed_path=config['processed_data_path']
    )

    # Initializing and training the model (Random Forest CLassifier)
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        random_state=config['random_state']
    )

    # Evaluating the model
    report = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Saving the trained model
    save_model(model,config['saved_model_path'])


if __name__ == "__main__":
    main()

    
