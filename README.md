# Air Filter Classifier

Classifying 11 industrial air-filter types with transfer learning (MobileNetV2 & EfficientNetB0). This repo highlights the approach, results, and key insights, with clear visuals and a concise code entrypoint.
---

## Highlights
- 11-class grayscale image classification (224×224)
- Pretrained backbones: MobileNetV2, EfficientNetB0
- Consistent preprocessing and evaluation
- Confusion matrices on three test sets (original, medium, hard)
- EarlyStopping and saved models per configuration

> <img width="582" height="523" alt="confusion_matrices_example" src="https://github.com/user-attachments/assets/e451ea8a-256e-4774-829d-d5a0d5395c85" />

Confusion matrix hard dataset

---

## Repository Structure
├── src/

│ └── main_training.py # training & evaluation script

│ ├── Project_report_Hruska.pdf

│ ├── Slides_air_filter_classifier.pptx

│ └── Classes_shown.pdf

└── README.md


## Approach

We use ImageNet-pretrained CNNs (MobileNetV2, EfficientNetB0) and a lightweight classification head. Grayscale images are replicated across RGB channels for compatibility. A small configuration grid explores fine-tuning depth, learning rate, and regularization.

## Results (example)

- EfficientNetB0 achieves near-perfect accuracy on the original test set.
- Medium and hard test sets expose robustness differences; EfficientNetB0 generally outperforms MobileNetV2.
- Over/under-exposure reduces discriminability between visually similar classes (notably 9 vs 10).

<img width="702" height="448" alt="results_barchart" src="https://github.com/user-attachments/assets/ce6affaf-aebb-45ff-8a04-907b7a4e51cc" />

Chart: accuracy per model across test sets  

<img width="648" height="669" alt="classes" src="https://github.com/user-attachments/assets/695114b7-97e3-4310-a508-07b0623c7c3f" />

Classes

---

## Key Learnings
- Strong baseline performance with lightweight transfer learning.
- Robustness requires mindful augmentation; extreme brightness/contrast harms fine details.
- Moderate fine-tuning and mild regularization often suffice for this dataset.

        print(f"Saved: {out_path}")
        for ds, tag in [(test_ds_orig, "Original"), (test_ds_med, "Medium"), (test_ds_hard, "Hard")]:
            loss, acc = model.evaluate(ds, verbose=0)
            print(f"{tag:8s} | loss={loss:.4f}  acc={acc:.4f}")
            plot_cm(model, ds, tag, class_names)
