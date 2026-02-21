import tensorflow as tf
import numpy as np
from PIL import Image
import os
# Silence TF logs (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Load Trained Model
model = tf.keras.models.load_model("cnn_model_regularized.keras")
# Load Image
img = Image.open("digit.png")
# Convert to grayscale
img = img.convert("L")
# Resize to 28x28
img = img.resize((28, 28))
# Convert to numpy array
img_array = np.array(img)
# Invert colors (MNIST format = white digit on black background)
img_array = 255 - img_array
# Normalize
img_array = img_array / 255.0
# Reshape to match model input
img_array = img_array.reshape(1, 28, 28, 1)
# Predict
logits = model.predict(img_array)
predicted_class = np.argmax(logits, axis=1)[0]
print("Predicted digit:", predicted_class)