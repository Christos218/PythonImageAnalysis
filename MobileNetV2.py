import os
import cv2
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Placeholder for the path to your Excel file and images directory
excel_path = "./Hotel_Dataset.xlsx"
images_directory = "./Images"

# Load the Excel file
data = pd.read_excel(excel_path)

# Preprocess and load images
images = [
    cv2.resize(cv2.imread(f"{images_directory}/{filename}"), (224, 224))
    for filename in data["new_img_name"]
]
images = np.array(images) / 255.0

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data["Food_Item"])
joblib.dump(label_encoder, "label_encoder.pkl")  # Save the LabelEncoder

num_classes = len(np.unique(encoded_labels))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    images, encoded_labels, test_size=0.2, random_state=42
)

# Load MobileNetV2 with pre-trained ImageNet weights, exclude top fully connected layers
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

# Freeze the layers of the base_model
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top of MobileNetV2
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)  # Add Dropout to prevent overfitting
predictions = Dense(num_classes, activation="softmax")(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
validation_datagen = ImageDataGenerator()  # No augmentation for validation data

# Apply data augmentation to the training data
train_datagen.fit(X_train)

# Splitting the training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

# Creating data generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
validation_generator = validation_datagen.flow(X_val, y_val, batch_size=32)

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

# Train the model with data augmentation
model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stop],
)

# Save the model in HDF5 format
model.save("my_model.h5")

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Re-compile the model for fine-tuning
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Continue training (fine-tuning)
model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stop],
)

# Save the fine-tuned model
model.save("my_finetuned_model.h5")

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average="weighted", zero_division=1)
recall = recall_score(y_test, y_pred_classes, average="weighted", zero_division=1)
f1 = f1_score(y_test, y_pred_classes, average="weighted", zero_division=1)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
