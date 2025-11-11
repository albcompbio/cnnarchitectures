import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, SGD

# Load and normalize MNIST data
mnist = tf.keras.datasets.mnist   #or cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = mnist.load_data()  #or cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define optimizers with proper names
optimizers = {
    "Adam": Adam(),
    "SGD": SGD(),
    "SGD with Momentum": SGD(learning_rate=0.01, momentum=0.9)
}

# Loop through each optimizer
for name, opt_instance in optimizers.items():
    print(f"\nUsing optimizer: {name}")

    # Build a fresh model for each optimizer
    model = Sequential([
        Flatten(input_shape=(28, 28)),    #or (32, 32, 3) for cifar10
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=opt_instance,
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()]
    )

    # Train the model
    model.fit(x_train, y_train, epochs=2, verbose=1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy with {name}: {test_acc:.4f}")
