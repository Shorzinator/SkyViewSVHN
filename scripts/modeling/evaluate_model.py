import numpy as np
# Import your data loading functions
from scripts.utility.data_loader import prepare_data_for_model
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model


def evaluate_model(model_path, X_test, y_test):
    """
    Load the trained model and evaluate it on the test set.

    Args:
        model_path (str): Path to the saved model.
        X_test: Test data.
        y_test: True labels for test data.

    Returns:
        dict: A dictionary containing various performance metrics.
    """
    # Load the saved model
    model = load_model(model_path)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Compute accuracy - You may need to adjust this based on your exact problem and output format
    accuracy = accuracy_score(np.argmax(y_test, axis=-1), np.argmax(y_pred, axis=-1))

    # More detailed performance analysis
    detailed_report = classification_report(np.argmax(y_test, axis=-1), np.argmax(y_pred, axis=-1))

    return {
        "accuracy": accuracy,
        "detailed_report": detailed_report
    }


if __name__ == "__main__":
    # Load and prepare test data
    data_dir = "path_to_your_data_directory"  # Update with your data directory
    _, _, X_test, y_test = prepare_data_for_model(...)  # Adjust as needed to load your test data

    # Evaluate the model
    model_path = "path_to_your_saved_model/svhn_model.h5"  # Update with your model path
    evaluation_metrics = evaluate_model(model_path, X_test, y_test)

    print("Model Accuracy:", evaluation_metrics["accuracy"])
    print("Detailed Performance Report:")
    print(evaluation_metrics["detailed_report"])
