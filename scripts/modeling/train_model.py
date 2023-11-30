# Import your data loading function
from scripts.utility.data_loader import load_svhn_data, prepare_data_for_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model_definition import create_cnn_rnn_model


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the CNN-RNN model.

    Args:
        model: Compiled CNN-RNN model.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        history: Training history object.
    """
    # Callbacks
    checkpoint_cb = ModelCheckpoint("svhn_model.h5", save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

    # Fit the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    return history


if __name__ == "__main__":
    # Load and prepare data
    data_dir = "path_to_your_data_directory"  # Update with your data directory
    images, bboxes = load_svhn_data(data_dir)
    X_train, X_val, y_train, y_val = prepare_data_for_model(images, bboxes)

    # Create model
    input_shape = X_train.shape[1:]  # Assuming X_train is a numpy array
    num_classes = 11  # 10 digits + 1 for 'no digit'
    model = create_cnn_rnn_model(input_shape, num_classes)

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val)
