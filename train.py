import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
# Load MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# Reshape & Normalize
#neural networks will get better performance upon scaling it
#computer has 8 bits so 2**8 = 256 : (0,255)
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0
# Build Model
model = tf.keras.Sequential([
    #28*28 pixel image and one channel (grayscale)
    tf.keras.Input(shape=(28, 28, 1)),
    # Conv Block 1
    #padding = 'same' means output will have same dimensions as input 28*28
    #kernel regulazier used for regularization and 0.0005 is constant
    layers.Conv2D(32, (3,3), padding="same",
                  kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2)),
    #dropout = 0.25 means 25% of neurons are turned off during training
    layers.Dropout(0.25),
    # Conv Block 2
    layers.Conv2D(64, (3,3), padding="same",
                  kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    # Dense Block
    layers.Flatten(),
    layers.Dense(256,
                 kernel_regularizer=regularizers.l2(0.0005)),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.5),
    # Output Layer
    #didits are (0-9) so 10 digits
    layers.Dense(10)  # logits
])
# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
# Callbacks (Early Stopping)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    #patience = 3 means if it for 3 iteration it doesn't show any improvments stop training
    patience=3,
    #When training stops, the model is currently at epoch 9 in our example but the best performance was at epoch 6.The model rewinds back to epoch 6 weights.
    restore_best_weights=True
)
# Train
model.fit(
    train_images,
    train_labels,
    epochs=2,
    validation_data=(test_images, test_labels),
    callbacks=[early_stop]
)
# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)
# Save Model
model.save("cnn_model_regularized.keras")
print("Model saved.")