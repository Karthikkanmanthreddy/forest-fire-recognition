# Smoke and Fire Recognition in Forest Images

Individual project, Summer 2026 module retake exam.
**Author:** Karthik Reddy Kanmanthreddy
**Module:** TODO (write the exact module name from your assignment)
**Report (Overleaf, view link):** TODO
**Code demo video:** TODO  |  **Presentation video:** TODO

## Task
Binary image classification: does a forest image show fire (`fire`) or not (`nofire`)?
Models: a from-scratch CNN baseline, and ImageNet-pretrained MobileNetV3-Small and ResNet18 fine-tuned.
Evaluation focus: fire-class recall, precision and F1, plus false-positive examples.

> These results describe image recognition on this particular dataset only.
> They do not show that a system can detect real wildfires.

## Dataset
Khan, A. and Hassan, B. (2020). *Dataset for Forest Fire Detection*. Mendeley Data, V1.
doi:[10.17632/gjmr63rz2r.1](https://doi.org/10.17632/gjmr63rz2r.1). Licence: CC BY 4.0.

The dataset has 1,900 RGB images of 250x250 pixels (950 per class), collected from web
search engines and cropped by the authors. **The images are not included in this repository.**
Download them from the DOI above and extract so that this layout exists:

```
data/raw/Dataset/
  Testing/fire, Testing/nofire                          (190 + 190 images)
  Training and Validation/fire, .../nofire              (760 + 760 images)
```

Changes made by this project: near-duplicate removal (below), a train/validation split,
and resizing to 224x224 at training time. No endorsement by the dataset authors is implied.

### Preparation and splits
`prepare_data.py` hashes every image (MD5 and 64-bit perceptual hash), groups near-duplicates
(Hamming distance <= 4) and removes 9 of 1,900 images: 8 duplicates inside the training
folder and 1 training image that duplicated a test image. It then splits the remaining
training-folder images into train/validation (stratified, group-aware, seed 42).
The authors' `Testing` folder is used unchanged as the locked test set.

| Split | fire | nofire | total |
|---|---|---|---|
| Train | 648 | 647 | 1,295 |
| Validation | 108 | 108 | 216 |
| Test (locked) | 190 | 190 | 380 |

The resulting split files are committed in `data/splits/`.

## Repository contents
| File | Purpose |
|---|---|
| `prepare_data.py` | scan, de-duplicate, split; writes `data/splits/*.csv` |
| `fire_common.py` | dataset, transforms, models, metrics, threshold selection |
| `train.py` | train one model; choose epoch and threshold on validation only |
| `evaluate.py` | one-time test evaluation, bootstrap CIs, figures, robustness test |
| `colour_baseline.py` | logistic regression on global colour statistics (diagnostic) |
| `results/` | metrics tables, test predictions, figures |
| `runs/` | training logs, curves, configs (checkpoints not committed) |

## Setup
Python 3.10+; a GPU is recommended for training (Google Colab T4 was used).
```
pip install -r requirements.txt
```

## Reproduce
```
python prepare_data.py --data-root data/raw/Dataset --out data/splits

python train.py --arch cnn                --epochs 30                       --target-recall 0.98 --name cnn_baseline
python train.py --arch mobilenet_v3_small --epochs 15 --freeze-epochs 3     --target-recall 0.98 --name mnv3_basic
python train.py --arch mobilenet_v3_small --epochs 15 --freeze-epochs 3 --aug strong --target-recall 0.98 --name mnv3_strong
python train.py --arch resnet18           --epochs 15 --freeze-epochs 3     --target-recall 0.98 --name r18_basic

python evaluate.py --runs cnn_baseline mnv3_basic mnv3_strong r18_basic --gallery-run r18_basic --robustness
python colour_baseline.py --also-test
```
Notes: epoch selection uses validation loss; the decision threshold is the strictest one
reaching 98% validation fire recall, capped at 0.5; seed 42. GPU training is not bit-for-bit
deterministic, so re-runs may differ slightly. `evaluate.py` touches the test set and was run
once after all training choices were fixed.

## Results (locked test set, n = 380; 95% bootstrap CIs)
Operating point: threshold fixed on validation.

| Model | Accuracy | Fire precision | Fire recall | Fire F1 | AP | FP | FN |
|---|---|---|---|---|---|---|---|
| CNN baseline (thr 0.418) | 0.932 [0.905, 0.955] | 0.914 [0.872, 0.950] | 0.953 [0.920, 0.979] | 0.933 [0.905, 0.957] | 0.972 | 17 | 9 |
| MobileNetV3-Small | 0.992 [0.982, 1.000] | 0.984 [0.965, 1.000] | 1.000 | 0.992 [0.982, 1.000] | 1.000 | 3 | 0 |
| MobileNetV3-Small, strong aug. | 0.989 [0.979, 0.997] | 0.979 [0.956, 0.995] | 1.000 | 0.990 [0.978, 0.998] | 0.999 | 4 | 0 |
| **ResNet18 (pre-selected primary)** | 0.987 [0.974, 0.997] | 0.974 [0.949, 0.995] | 1.000 | 0.987 [0.974, 0.997] | 0.999 | 5 | 0 |

Diagnostic baselines, added after the main experiments (logistic regression, test set):
brightness only 0.658 accuracy (AUC 0.728); mean R,G,B 0.863 accuracy (AUC 0.937).

Main observations: transfer learning clearly beats the from-scratch CNN; the three transfer
models are statistically indistinguishable; ResNet18 was designated primary by a rule fixed
before the test run (lowest validation loss among transfer models), although MobileNetV3
scored slightly higher on test. Robustness results (brightness, blur, resolution, JPEG) are
in `results/robustness.csv` and `results/figures/robustness.png`.

## Limitations
- Small, web-scraped dataset with tightly cropped fire images; class colour and brightness
  differ, and a colour-only classifier already reaches about 86% test accuracy.
- Validation was easier than the test folder, so validation scores are optimistic.
- One training run (seed) per model; architecture differences are within noise.
- Robustness perturbations are synthetic and thresholds were not re-calibrated under shift.
- Not evidence of real-world wildfire detection performance.

## AI-use statement
TODO: describe honestly which tools you used and for what (for example, an AI assistant
helped scaffold and debug the code; the report text is written by me). Keep it consistent
with the statement in the report.

## Credits
Dataset: Khan and Hassan (2020), CC BY 4.0. Pretrained weights: torchvision ImageNet models.
Libraries: PyTorch, torchvision, scikit-learn, pandas, matplotlib, Pillow, SciPy.
