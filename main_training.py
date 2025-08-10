import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os

# Nastavení cest
base_data_dir = "C:/Project_air_filter/Typy_filtru/"
data_dir = os.path.join(base_data_dir, "dataset_preprocessed_augmented_larger_enhanced")
test_dir_1 = os.path.join(base_data_dir, "test_data_augmented")
test_dir_2 = os.path.join(base_data_dir, "test_data_original")
test_dir_3 = os.path.join(base_data_dir, "test_data_augmented_hard")

# Načtení dat
batch_size = 32
img_size = (224, 224)

def load_dataset(directory):
    return image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        seed=123
    )

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale"
)
val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale"
)

# Testovací sady
test_ds_1 = load_dataset(test_dir_1)
test_ds_2 = load_dataset(test_dir_2)
test_ds_3 = load_dataset(test_dir_3)

# Funkce pro sestavení modelu
def build_model(base_model, trainable_layers, learning_rate, dropout_rate, use_l2):
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    regularizer = l2(0.01) if use_l2 else None

    model = Sequential([
        tf.keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x), input_shape=(224, 224, 1)),
        base_model,
        GlobalAveragePooling2D(),
        Dropout(dropout_rate),
        Dense(128, activation='relu', kernel_regularizer=regularizer),
        Dense(11, activation='softmax', kernel_regularizer=regularizer)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Funkce pro vykreslení confusion matrix
def plot_confusion_matrix(model, dataset, dataset_name):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_ds.class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.show()

# Iterace přes kombinace
configs = [
    {"trainable_layers": 0, "learning_rate": 1e-4, "dropout_rate": 0.2, "use_l2": False},
    {"trainable_layers": 5, "learning_rate": 1e-4, "dropout_rate": 0.2, "use_l2": False},
    {"trainable_layers": 10, "learning_rate": 1e-4, "dropout_rate": 0.2, "use_l2": False},
    {"trainable_layers": 15, "learning_rate": 1e-4, "dropout_rate": 0.2, "use_l2": False},
    {"trainable_layers": 0, "learning_rate": 1e-5, "dropout_rate": 0.4, "use_l2": True},
    {"trainable_layers": 5, "learning_rate": 1e-5, "dropout_rate": 0.4, "use_l2": True},
    {"trainable_layers": 10, "learning_rate": 1e-5, "dropout_rate": 0.4, "use_l2": True},
    {"trainable_layers": 15, "learning_rate": 1e-5, "dropout_rate": 0.4, "use_l2": True},
]

models = {"MobileNetV2": MobileNetV2, "EfficientNetB0": EfficientNetB0}

for model_name, model_function in models.items():
    for config in configs:
        print(f"Training {model_name} with config: {config}")

        base_model = model_function(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        model = build_model(
            base_model,
            config["trainable_layers"],
            config["learning_rate"],
            config["dropout_rate"],
            config["use_l2"]
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            ]
        )

        model.save(f"C:/Project_air_filter/Typy_filtru/{model_name}_layers{config['trainable_layers']}_lr{config['learning_rate']}_dropout{config['dropout_rate']}.keras")

        for test_ds, name in zip([test_ds_1, test_ds_2, test_ds_3], ["Augmented Test Data", "Original Test Data", "Hard Test Data"]):
            test_loss, test_accuracy = model.evaluate(test_ds)
            print(f"{name} - Loss: {test_loss}, Accuracy: {test_accuracy}")
            plot_confusion_matrix(model, test_ds, name)
