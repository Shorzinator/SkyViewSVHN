from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, LSTM, MaxPooling2D
from tensorflow.keras.model import Model


def create_cnn_rnn_model(input_shape, num_classes, max_digits=5):
    """
    Create a CNN-RNN model for sequence recognition.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): Number of classes (10 digits + 'no digit' class).
        max_digits (int): Maximum length of digit sequences.

    Returns:
        model: A compiled Keras model.
    """
    # Input layer
    inputs = Input(shape=input_shape)

    # CNN layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten and apply dropout
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    # RNN layers
    x = Dense(max_digits * 128)(x)  # Prepare vector for reshaping into (max_digits, 128)
    x = Reshape((max_digits, 128))(x)
    x = LSTM(128, return_sequences=True)(x)

    # Output layer for each digit position
    outputs = [Dense(num_classes, activation='softmax', name=f'digit_{i + 1}')(x) for i in range(max_digits)]

    # Define and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    input_shape = (64, 64, 3)  # Example input shape, adjust as needed
    num_classes = 11  # 10 digits + 1 for 'no digit'
    model = create_cnn_rnn_model(input_shape, num_classes)
    model.summary()
