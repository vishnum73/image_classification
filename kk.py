# Create Dataset
#       ↓
# Install TensorFlow
#       ↓
# Load Images
#       ↓
# Data Augmentation
#       ↓
# Load MobileNetV2
#       ↓
# Freeze Base Model
#       ↓
# Add Classification Layers
#       ↓
# Compile Model
#       ↓
# Train Model
#       ↓
# Save Model
#       ↓
# Load Model
#       ↓
# Predict New Images


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Settings
# -----------------------------
img_height = 224
img_width = 224
batch_size = 32
epochs = 10

# -----------------------------
# Load Dataset
# Folder Structure:
# dataset/
#    cat/
#    dog/
# -----------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_dataset.class_names
if len(class_names) < 2:
    raise ValueError(
        "The training dataset must contain at least 2 class subfolders (for example 'cat' and 'dog'). "
        f"Found: {class_names}"
    )
print("Classes:", class_names)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# Data Augmentation
# -----------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

# -----------------------------
# Load MobileNetV2
# -----------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# -----------------------------
# Build Model
# -----------------------------

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

# -----------------------------
# Compile Model
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Model Summary
# -----------------------------

model.summary()

# -----------------------------
# Train Model
# -----------------------------

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)

# -----------------------------
# Evaluate Model
# -----------------------------

loss, accuracy = model.evaluate(validation_dataset)

print("Validation Accuracy:", accuracy)

# -----------------------------
# Save Model
# -----------------------------

model.save("image_classifier.keras")

print("Model Saved Successfully")

# -----------------------------
# Predict One Image
# -----------------------------

import os

# Find a sample test image in the workspace test folder
test_image_path = None
for root, _, files in os.walk("test"):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            test_image_path = os.path.join(root, file)
            break
    if test_image_path:
        break

if not test_image_path or not os.path.exists(test_image_path):
    print("Error: No test image found in the 'test' folder.")
else:
    print(f"Predicting on image: {test_image_path}")
    try:
        image = tf.keras.utils.load_img(
            test_image_path,
            target_size=(img_height, img_width)
        )

        image_array = tf.keras.utils.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)

        prediction = model.predict(image_array)

        score = tf.nn.softmax(prediction[0])

        print("Predicted Class:", class_names[np.argmax(score)])
        print("Confidence:", round(100 * np.max(score), 2), "%")
    except Exception as e:
        print(f"Error loading or predicting image: {str(e)}")

# -----------------------------
# Plot Accuracy
# -----------------------------

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])
plt.show()

# -----------------------------
# Plot Loss
# -----------------------------

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train","Validation"])
plt.show()