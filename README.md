# Air Filter Classifier — Reproducible Project

A fully reproducible TensorFlow/Keras project for classifying 11 industrial air-filter types using transfer learning (MobileNetV2 & EfficientNetB0). This single README provides:
- Exact folder structure to create.
- What files and images to place where.
- Copy‑paste code for training, evaluation, and plotting confusion matrices.
- Clear image placeholders so the repository renders nicely on GitHub.

If cloning this as-is, follow the “Folder Structure to Create” and “What to Put Where” sections first.

---

## 1) Folder Structure to Create

Create this exact structure in the repo root:

text
# Air Filter Classifier — Reproducible Project

A fully reproducible TensorFlow/Keras project for classifying 11 industrial air-filter types using transfer learning (MobileNetV2 & EfficientNetB0). This single README provides:
- Exact folder structure to create.
- What files and images to place where.
- Copy‑paste code for training, evaluation, and plotting confusion matrices.
- Clear image placeholders so the repository renders nicely on GitHub.

If cloning this as-is, follow the “Folder Structure to Create” and “What to Put Where” sections first.

---

## 1) Folder Structure to Create

Create this exact structure in the repo root:

.
├── data/
│ ├── dataset_preprocessed_augmented_larger_enhanced/ # TRAIN+VAL source (augmented)
│ │ ├── class_1/ … class_11/ # Put training images here (see below)
│ ├── test_data_original/ # TEST set (original)
│ │ ├── class_1/ … class_11/ # Put original test images here
│ ├── test_data_augmented/ # TEST set (medium difficulty)
│ │ ├── class_1/ … class_11/ # Put medium-aug test images here
│ └── test_data_augmented_hard/ # TEST set (hard difficulty)
│ ├── class_1/ … class_11/ # Put hard-aug test images here
├── images/ # Repo visuals used in this README
│ ├── banner_classes_grid.png
│ ├── confusion_matrices_example.png
│ ├── pipeline_diagram.png
│ ├── class9_vs_class10_glue.png
│ └── results_barchart.png
├── reports/
│ ├── Project_report_Hruska.pdf
│ ├── Slides_air_filter_classifier.pptx
│ └── Classes_shown.pdf
├── src/
│ └── main_training.py # Copy the code from this README
├── models/ # Auto-created at runtime (saved models)
├── requirements.txt
└── README.md



Notes:
- All image files used in this README go in images/.
- All code goes in src/main_training.py.
- The models/ folder will be created automatically by the script if it does not exist.

---

## 2) What to Put Where

Images for training/testing (these are the images the model learns from):
- Place training/validation images into:
  - data/dataset_preprocessed_augmented_larger_enhanced/class_1 … class_11
  - Each class folder should contain .png or .jpg images sized 224×224, grayscale if possible.
- Place test images into:
  - data/test_data_original/class_1 … class_11
  - data/test_data_augmented/class_1 … class_11
  - data/test_data_augmented_hard/class_1 … class_11

Minimum requirement to run:
- At least a handful of images per class in each test set, but for meaningful results target ≥100/class in train, and ≥50/class in test.

Images for README visuals (for GitHub presentation; not used for training):
- images/banner_classes_grid.png
  - A collage showing one sample from each of the 11 classes (suggested).
- images/confusion_matrices_example.png
  - A stitched image of three confusion matrices (original/medium/hard) from a previous run; use placeholders until you have results.
- images/pipeline_diagram.png
  - A simple diagram “Data → Training → Evaluation”.
- images/class9_vs_class10_glue.png
  - Two sample images illustrating the glue stripe difference between classes 9 and 10.
- images/results_barchart.png
  - A bar chart summarizing accuracies per model/config (optional placeholder until you have results).

Reports (optional, for documentation):
- Place any PDFs/PowerPoints (e.g., Project_report_Hruska.pdf, Slides_air_filter_classifier.pptx, Classes_shown.pdf) into reports/.

---

## 3) README Visual Placeholders (GitHub shows these nicely)

> Image placeholder  
> A banner or collage of the 11 classes  
> Path: images/banner_classes_grid.png

> Image placeholder  
> Three confusion matrices (original / medium / hard)  
> Path: images/confusion_matrices_example.png

> Image placeholder  
> Side-by-side of class 9 vs class 10 highlighting glue stripe  
> Path: images/class9_vs_class10_glue.png

> Image placeholder  
> Pipeline: data → training → evaluation  
> Path: images/pipeline_diagram.png

> Image placeholder  
> Bar chart of test accuracies across datasets  
> Path: images/results_barchart.png

---

## 4) Environment Setup

Create requirements.txt in the repo root with:

