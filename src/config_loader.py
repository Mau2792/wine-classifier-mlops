import yaml
from typing import Dict, Any

def load_config(config_path: str="config.yaml") -> Dict[str, Any]:
    """
    Loads the configuration at the specific path passed as argument.
    Returns the configuration as a dictionary
    """
    try:
        # Loading the configuration using yaml safe_load method
        with open (config_path, "r") as file:
            config = yaml.safe_load(file)

        print(f"{config_path} successfully loaded")
        return config
    
    except FileNotFoundError:
        print(f"ERROR: File {config_path} not found")
        raise

    except yaml.YAMLError:
        print(f"ERROR: Error reading the file {config_path}: {error}")
        raise



