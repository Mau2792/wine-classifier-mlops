from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from typing import Dict

def evaluate_model(model: any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates the model and returns accuracy score, prints a detailed report using sklearn classification_report function.
    ----------------------------------------
    Args:
    - model, the model to evaluate
    - X_test, test dataset used for prediction
    - t_test, X_test attended results
    ----------------------------------------
    """
    # Testing the model on X_test dataset
    predictions = model.predict(X_test)

    # Calculating accuracy with accuracy_score by sklearn
    accuracy = accuracy_score(y_pred=predictions, y_true=y_test)

    print(f"The model as an  accuracy of {accuracy*100:.2f}%")

    # Printing a detailed report using classification_report by sklearn
    report = classification_report(
        y_pred=predictions,
        y_true=y_test,
        target_names=["Class 0", "Class 1", "Class 2"]
    )
    print("CLASSIFICATION REPORT:")
    print(report)

    return {"Accuracy: ", accuracy}


