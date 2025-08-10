# Air Filter Classifier

Classifying 11 industrial air-filter types with transfer learning (MobileNetV2 & EfficientNetB0). This repo highlights the approach, results, and key insights, with clear visuals and a concise code entrypoint.

> Image placeholder  
> Project banner / collage (1 sample per class)  
> Path: images/banner_classes_grid.png

---

## Highlights
- 11-class grayscale image classification (224×224)
- Pretrained backbones: MobileNetV2, EfficientNetB0
- Consistent preprocessing and evaluation
- Confusion matrices on three test sets (original, medium, hard)
- EarlyStopping and saved models per configuration

> Image placeholder  
> Three confusion matrices (original / medium / hard)  
> Path: images/confusion_matrices_example.png

---

## Repository Structure
├── src/

│ └── main_training.py # training & evaluation script

├── models/ # saved .keras models (created when you run)

├── images/ # visuals for the README (placeholders below)

│ ├── banner_classes_grid.png

│ ├── confusion_matrices_example.png

│ ├── pipeline_diagram.png

│ ├── class9_vs_class10_glue.png

│ └── results_barchart.png

├── reports/ # optional docs (if you want to include)

│ ├── Project_report_Hruska.pdf

│ ├── Slides_air_filter_classifier.pptx

│ └── Classes_shown.pdf

└── README.md



If you’d like to run the code, create your own data folders under data/... following a standard Keras directory layout. For this “showcase” repo, we only include code and visuals.

---

## Approach

We use ImageNet-pretrained CNNs (MobileNetV2, EfficientNetB0) and a lightweight classification head. Grayscale images are replicated across RGB channels for compatibility. A small configuration grid explores fine-tuning depth, learning rate, and regularization.

> Image placeholder  
> Pipeline diagram: Data → Preprocess → Train → Evaluate  
> Path: images/pipeline_diagram.png

---

## Results (example)

- EfficientNetB0 achieves near-perfect accuracy on the original test set.
- Medium and hard test sets expose robustness differences; EfficientNetB0 generally outperforms MobileNetV2.
- Over/under-exposure reduces discriminability between visually similar classes (notably 9 vs 10).

> Image placeholder  
> Bar chart: accuracy per model across test sets  
> Path: images/results_barchart.png

> Image placeholder  
> Class 9 vs Class 10 (glue stripe highlight)  
> Path: images/class9_vs_class10_glue.png

---

## Key Learnings
- Strong baseline performance with lightweight transfer learning.
- Robustness requires mindful augmentation; extreme brightness/contrast harms fine details.
- Moderate fine-tuning and mild regularization often suffice for this dataset.

---

## Code (src/main_training.py)

The script below trains and evaluates MobileNetV2 and EfficientNetB0 variants, saving models and plotting confusion matrices. If you want to actually run it, add your data under data/... using the Keras directory convention.

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

--------- Config ---------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(file), ".."))
DATA_DIR_TRAIN = os.path.join(BASE_DIR, "data", "dataset_preprocessed_augmented_larger_enhanced")
DATA_DIR_TEST_ORIG = os.path.join(BASE_DIR, "data", "test_data_original")
DATA_DIR_TEST_MED = os.path.join(BASE_DIR, "data", "test_data_augmented")
DATA_DIR_TEST_HARD = os.path.join(BASE_DIR, "data", "test_data_augmented_hard")
SAVE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
NUM_CLASSES = 11
SEED = 123

--------- Data loaders (only used if you add data) ---------
def load_dataset(directory, shuffle=False):
return image_dataset_from_directory(
directory,
image_size=IMG_SIZE,
batch_size=BATCH_SIZE,
color_mode="grayscale",
seed=SEED,
shuffle=shuffle
)

def try_load_data():
if not (os.path.isdir(DATA_DIR_TRAIN) and
os.path.isdir(DATA_DIR_TEST_ORIG) and
os.path.isdir(DATA_DIR_TEST_MED) and
os.path.isdir(DATA_DIR_TEST_HARD)):
print("Data folders not found. This repository is set up as a showcase.")
print("Add data/ folders if you want to train/evaluate.")
return None
train_ds = image_dataset_from_directory(
DATA_DIR_TRAIN, validation_split=0.2, subset="training", seed=SEED,
image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale")
val_ds = image_dataset_from_directory(
DATA_DIR_TRAIN, validation_split=0.2, subset="validation", seed=SEED,
image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale")
class_names = train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds_med = load_dataset(DATA_DIR_TEST_MED)
test_ds_orig = load_dataset(DATA_DIR_TEST_ORIG)
test_ds_hard = load_dataset(DATA_DIR_TEST_HARD)
return train_ds, val_ds, test_ds_orig, test_ds_med, test_ds_hard, class_names

--------- Model builder ---------
def build_model(base_model, trainable_layers, learning_rate, dropout_rate, use_l2):
base_model.trainable = True
if trainable_layers > 0:
for layer in base_model.layers[:-trainable_layers]:
layer.trainable = False
else:
for layer in base_model.layers:
layer.trainable = False
regularizer = l2(0.01) if use_l2 else None
model = Sequential([
tf.keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x),
input_shape=(IMG_SIZE, IMG_SIZE, 1)),
base_model,
GlobalAveragePooling2D(),
Dropout(dropout_rate),
Dense(128, activation='relu', kernel_regularizer=regularizer),
Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizer)
])
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)
return model

--------- CM plotting ---------
def plot_cm(model, dataset, name, class_names):
y_true, y_pred = [], []
for images, labels in dataset:
preds = model.predict(images, verbose=0)
y_true.extend(labels.numpy())
y_pred.extend(np.argmax(preds, axis=1))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax, colorbar=False)
plt.title(f"Confusion Matrix — {name}")
plt.tight_layout()
plt.show()

--------- Training grid ---------
configs = [
{"trainable_layers": 0, "learning_rate": 1e-4, "dropout_rate": 0.2, "use_l2": False},
{"trainable_layers": 10, "learning_rate": 1e-4, "dropout_rate": 0.2, "use_l2": False},
{"trainable_layers": 0, "learning_rate": 1e-5, "dropout_rate": 0.4, "use_l2": True},
{"trainable_layers": 10, "learning_rate": 1e-5, "dropout_rate": 0.4, "use_l2": True},
]
backbones = {"MobileNetV2": MobileNetV2, "EfficientNetB0": EfficientNetB0}

--------- Main ---------
if name == "main":
data_bundle = try_load_data()
if data_bundle is None:
# Showcase mode: nothing to run if no data present.
exit(0)

train_ds, val_ds, test_ds_orig, test_ds_med, test_ds_hard, class_names = data_bundle

for name, backbone in backbones.items():
    for cfg in configs:
        print(f"\nTraining {name} | cfg={cfg}")
        base = backbone(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                        include_top=False, weights="imagenet")
        model = build_model(
            base_model=base,
            trainable_layers=cfg["trainable_layers"],
            learning_rate=cfg["learning_rate"],
            dropout_rate=cfg["dropout_rate"],
            use_l2=cfg["use_l2"]
        y
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            )
        ]
        model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks, verbose=1)
        out_path = os.path.join(
            SAVE_DIR,
            f"{name}_L{cfg['trainable_layers']}_lr{cfg['learning_rate']}_do{cfg['dropout_rate']}.keras"
        )
        model.save(out_path)
        print(f"Saved: {out_path}")
      
        for ds, tag in [(test_ds_orig, "Original"), (test_ds_med, "Medium"), (test_ds_hard, "Hard")]:
            loss, acc = model.evaluate(ds, verbose=0)
            print(f"{tag:8s} | loss={loss:.4f}  acc={acc:.4f}")
            plot_cm(model, ds, tag, class_names)
