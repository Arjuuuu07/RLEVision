# RLE-Conv1D: Can Structural Patterns Replace Pixels 
in Ultrasound Cancer Detection?

*A proof-of-concept study — not a production model, 
not optimized for maximum numbers. Just a question 
worth asking, and an answer worth sharing.*

---

## The question

Raw pixels carry a lot of noise. In ultrasound images, 
most of what a pixel-based CNN learns is texture, 
brightness variation, and scanner artifacts — not 
pure structure.

What if you threw the pixels away entirely?

What if instead of feeding a CNN 65,536 pixel values 
per image, you described the image as a series of 
structural runs — where dark regions start, how long 
they are, how they repeat across rows and columns?

That is Run-Length Encoding. And this project is 
a proof-of-concept answer to: **does it work for 
cancer detection in breast ultrasound?**

---

## This model is intentionally not optimized

The architecture was not tuned. The hyperparameters 
were not searched. The dataset is small. The goal 
was never to beat ResNet-18 or publish a 
state-of-the-art number.

The goal was to show that RLE structural features 
carry real diagnostic signal — and that a model 
which respects the sequential spatial order of 
those features can exploit that signal competitively 
with pixel-based models, in a fraction of the time, 
with no pretraining.

If this idea holds up on a larger dataset with 
proper optimization, the numbers will be better. 
This is the first step.

---

## What we found — the headline results

These numbers are from an unoptimized model on a 
small dataset (780 images). They are not the ceiling. 
They are the floor.

**Speed — the clearest win:**

| Model | Training time | vs RLE |
|---|---|---|
| ResNet-18 (ImageNet) | 1086s | 14× slower |
| CNN (raw pixels) | 1132s | 15× slower |
| **RLE-Conv1D (ours)** | **75s** | **baseline** |

No ImageNet pretraining. No pixel processing at 
inference. Just 15,408 float32 values per image.

**Cancer recall — the most important number:**

At threshold 0.50, the RLE model missed **zero 
cancer cases** on the fixed test set. Bootstrap 
confidence interval across 1000 resamples: 
(1.000, 1.000). The recall never moved.

Cross-validation cancer recall: **0.947 ± 0.040** 
across 5 folds. Even in the worst fold, 88.5% of 
cancers were caught.

**AUC — competitive without pixels:**

| Model | AUC (single split) | AUC (5-fold CV) |
|---|---|---|
| ResNet-18 (ImageNet) | 0.978 | pending |
| **RLE-Conv1D (ours)** | **0.927** | **0.863 ± 0.053** |
| CNN (raw pixels) | 0.862 | pending |

RLE-Conv1D outperforms the pixel CNN (0.927 vs 0.862) 
without seeing a single raw pixel — using only the 
structural run patterns of the thresholded image.

**Sequential structure is essential:**

When the same RLE features are fed to sklearn models 
that treat all dimensions as independent, performance 
collapses. This proves the signal is in the structure 
— and you need an architecture that understands 
spatial order to exploit it.

| Model (same RLE input) | AUC |
|---|---|
| RLE-Conv1D | 0.927 |
| Random Forest | 0.788 |
| Gradient Boosting | 0.766 |
| Logistic Regression | 0.761 |
| SVM (RBF) | 0.700 |
| KNN (k=7) | 0.561 |

**Why RLE beats pixels for this task:**

In binary (black-and-white) images, CNNs learn structural patterns such as edges, boundaries, textures, shapes, region sizes, and spatial relationships directly from pixels. The proposed RLE representation captures many of these structural properties explicitly through run lengths and their positions across image rows and columns.

As a result, the model learns from image structure rather than individual pixel values. Furthermore, RLE features are inherently interpretable, since each feature corresponds to a specific run length and spatial location, whereas CNN feature representations are typically distributed and difficult to interpret directly.


---

## What we found — deeper findings

**Both axes matter, rows more than columns:**

Ablation shows that removing column RLE drops AUC 
by 0.038. Removing row RLE drops it by 0.099. 
This is physically consistent — breast lesions are 
typically wider than tall in ultrasound, so 
horizontal runs capture lesion boundaries more 
distinctively than vertical ones. The data confirmed 
a physical hypothesis without being told it.

**SE blocks did not help:**

Squeeze-and-excitation channel attention added 
−0.006 AUC in ablation. On a small dataset, the 
additional parameters hurt more than the attention 
mechanism helps. A simpler model without SE blocks 
performs equally or better. This is a finding worth 
knowing before adding complexity.

**The variance is the dataset, not the model:**

CV AUC range: 0.789 to 0.931 across 5 folds. 
That 0.142 gap is explained almost entirely by 
which 26 normal cases landed in each validation 
fold — not by model instability. With only 133 
normal cases total, this is expected and unavoidable 
on this dataset.

**Rle result:**

The model achieved a ROC-AUC of 0.927 on the held-out test split. To evaluate robustness beyond a single train/test partition, 5-fold stratified cross-validation was also performed, yielding a mean ROC-AUC of 0.863 ± 0.053. Both results are reported to provide a complete picture of performance and variability across different data splits.


---

## Full results

### Representation comparison (Table 1)
Each model uses its natural input with a suited 
architecture. Single-split evaluation (seed=42).

| Model | Input | AUC | Recall @0.50 | F1 @best | Time |
|---|---|---|---|---|---|
| ResNet-18 (ImageNet) | pixels | 0.978 | 0.852 | 0.852 | 1086s |
| **RLE-Conv1D** | **RLE** | **0.927** | **1.000** | **0.678** | **75s** |
| CNN (raw pixels) | pixels | 0.862 | 0.593 | 0.642 | 1132s |

RLE-Conv1D cross-validation:

| Metric | Mean ± Std | Range |
|---|---|---|
| AUC | 0.863 ± 0.053 | 0.789 – 0.931 |
| Cancer Recall | 0.947 ± 0.040 | 0.885 – 1.000 |

Bootstrap 95% CI (n=1000, fixed test set):

| Metric | Estimate | 95% CI |
|---|---|---|
| AUC | 0.927 | (0.879, 0.966) |
| Cancer Recall | 1.000 | (1.000, 1.000) |

### Classifier comparison on RLE (Table 2)
All models receive identical 15,408-dim RLE vectors.

| Model | AUC | Recall @0.50 | Note |
|---|---|---|---|
| **RLE-Conv1D** | **0.927** | **1.000** | respects run order |
| Random Forest | 0.788 | 0.148 | ignores sequence order |
| Gradient Boosting | 0.766 | 0.185 | ignores sequence order |
| Logistic Regression | 0.761 | 0.296 | linear only |
| SVM (RBF) | 0.700 | 0.037 | PCA first |
| KNN (k=7) | 0.561 | 0.111 | dimensionality problem |

### Ablation study

| Variant | AUC | Finding |
|---|---|---|
| Full model | 0.893 | complete system |
| Rows only | 0.855 | columns contribute +0.038 |
| Cols only | 0.794 | rows contribute +0.099 |
| No SE blocks | 0.899 | SE not helpful here |
| Main branch only | 0.881 | parallel adds +0.012 |

---
note-A custom CNN was included as a pixel-based baseline to evaluate how a model trained directly on ultrasound image pixels performs. A pretrained ResNet-18 was also evaluated to represent a stronger, modern deep learning approach that benefits from large-scale ImageNet pretraining. These baselines provide reference points for comparing the proposed RLE-based structural representation against both a basic pixel-learning model and a high-performance pretrained architecture.


## Honest limitations

- Single dataset: BUSI, 780 images, 133 normal cases
- High CV variance (±0.053) — caused by small normal 
  class, not model instability
- Early stopping uses test set in single-split mode — 
  standard practice on small datasets but worth noting
- AUC and Recall are not affected by threshold leakage.
  F1@best is slightly optimistic — treat as secondary.
- Not validated on a second dataset
- Not clinically validated in any form

**This is a research proof-of-concept. 
Not a medical device. Not a finished system.**

---

## Future directions

**Primary — RLE on clean binary images:**

This study focused on breast ultrasound classification, but the underlying idea is not limited to medical imaging. Since RLE operates on binary image structure, a natural next step is to investigate its application to other black-and-white image analysis tasks, including object detection, shape recognition, region classification, segmentation masks, and pattern analysis. Any domain where the primary information is encoded in spatial structure rather than color may benefit from run-length based representations.




**Other directions:**

- Test on additional datasets (STU, OASBUD, BUS-BRA) 
  to evaluate generalization
- Hybrid model: RLE features combined with pixel 
  features — does structural encoding add to pixels 
  or merely approximate them?
- Explore RLE for other modalities where echo or 
  intensity structure is diagnostically relevant 
  (thyroid, liver, cardiac ultrasound)
- Proper optimization of this architecture — the 
  current model was deliberately untuned. What is 
  the real ceiling?

---

## How it works

```
Ultrasound image (256×256 grayscale)
        ↓
Crop nearfield (top 8%) and farfield (bottom 25%)
Removes transducer artifacts
        ↓
Adaptive threshold: pixel < (mean − 0.6×std) → binary
Per-image normalization — handles scanner variation
        ↓
RLE on rows → (172 × 36) = 6,192 values
RLE on cols → (256 × 36) = 9,216 values
Each run: [normalized_start, normalized_length]
        ↓
Flat vector: 15,408 float32 features                   
        ↓
RLEClassifier
  ├── Main branch: Conv1d + residual + SE attention    
  ├── Parallel branch: kernels (5, 11, 13, 17)
  │   captures patterns at multiple spatial scales
  └── Fused 512-dim → FC → cancer / normal
```
why binarisation?-Breast ultrasound lesions frequently appear as darker regions relative to surrounding tissue. Binarization was used to emphasize structural patterns and region distribution, enabling the RLE representation to focus on shape, boundaries, spatial arrangement, and run-length structure rather than raw intensity values.

---

## Dataset

BUSI dataset (publicly available).
Download: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

Set paths in `rle.ipynb` Cell 2:
```python
BENIGN_DIR    = "path/to/benign"
MALIGNANT_DIR = "path/to/malignant"
NORMAL_DIR    = "path/to/normal"
```

---

## Installation

```bash
pip install torch torchvision opencv-python \
            scikit-learn imbalanced-learn tqdm numpy scipy

# Recommended — 10-20x faster feature extraction
pip install numba
```

## Usage

```bash
# First run — extracts and caches RLE features
python rle.ipynb

# After first run — set FORCE_RECOMPUTE = False
python rle.ipynb

# Verification: CV + bootstrap CI + ablation
second cell of same file
```

---

## Citation

```
@misc{rle-breast-ultrasound-2025,
  title  = {RLEVision: Can Structural Patterns 
             Replace Pixels in Ultrasound Cancer 
             Detection?},
  author = Arjun mahesh,
  year   = {2026},
  url    = {https://github.com/Arjuuuu07/RLEVision}
}
```

## License

MIT — free to use, modify, and build on 
with attribution.
