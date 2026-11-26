# Concept Bottleneck Model vs. direct model on SkinCon

This folder contains our report and a notebook that reproduces every number and figure in it.

In dermatology, a model that only outputs a diagnosis is hard to check. A Concept Bottleneck Model (CBM) first predicts clinical concepts that a dermatologist would recognise, such as erythema, plaque or ulcer, and then bases the diagnosis on those concepts alone. We compare such a model with a direct model that goes straight from the image to the diagnosis, and ask:

- **Performance:** how much accuracy does the concept bottleneck cost?
- **Q1:** does the direct model, never trained on concepts, still encode them?
- **Q2:** does the direct model also rely on information outside the annotated concepts?

## Setup

- **Data:** the 3,230 usable SkinCon images of Fitzpatrick17k, with 22 dermatologist-annotated concepts. There are two diagnostic targets, with 9 and with 3 classes.
- **Backbone:** frozen DINOv3 ViT-L/16. Both models start from the same CLS features.
- **Models:** both models have a 128-unit hidden layer. The direct model maps it to the diagnosis. The sequential CBM maps it to the 22 concepts and then predicts the diagnosis with a linear layer on those concepts.
- **Evaluation:** 5-fold stratified cross-validation with identical folds and seeds for both models.
- **Analyses:** linear concept probes, SHAP values over the concepts, and Grad-CAM on block 21 of the ViT.

## Main results

| Task | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| 9 classes | Direct | 0.751 ± 0.017 | 0.529 ± 0.040 |
| | Sequential CBM | 0.713 ± 0.027 | 0.509 ± 0.039 |
| 3 classes | Direct | 0.817 ± 0.012 | 0.710 ± 0.023 |
| | Sequential CBM | 0.811 ± 0.023 | 0.702 ± 0.029 |

- On the 3-class task the CBM matches the direct model. On the 9-class task it loses about 4 points of accuracy.
- The CBM recognises the concepts well (mean AUROC 0.88).
- A linear probe recovers all 22 concepts from the direct model's hidden layer (mean AUROC 0.82). Most of this information is already present in the frozen DINOv3 features (0.85).
- On average, 51% of the direct model's Grad-CAM evidence falls outside the regions of the annotated concepts.

The report discusses what this means for interpretability and for clinical use.

## Contents

| File | Description |
|---|---|
| `report.pdf` | The report. |
| `direct_vs_cbm.ipynb` | The notebook. It follows the report section by section, and its outputs are saved, so it can be read without running it. |
| `fitzpatrick17k_concepts.csv` | SkinCon concept annotations: the 48 concepts marked per image. |
| `fitzpatrick17k_vit_large_patch16_dinov3_lvd1689m_cls_float32.npy` | Precomputed DINOv3 CLS features for all 16,577 Fitzpatrick17k images. If deleted, the notebook recomputes them, which takes about 15 minutes on a GPU. |
| `requirements.txt` | Python packages, with the versions we tested. |

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook direct_vs_cbm.ipynb
```

Run all cells, and keep the notebook in this folder because it reads the two data files from here. The first run downloads two things:

- the Fitzpatrick17k images and metadata from Kaggle, a few GB, cached in `data/`;
- the DINOv3 ViT-L weights through timm, about 1.2 GB.

Sections 1–6 train the models and only need the precomputed features. They take a few minutes, even on a CPU. Sections 7–9 run the full ViT on more than 200 images and need a GPU. On a 6 GB laptop GPU the whole notebook runs in about 15 minutes and peaks at about 5 GB of RAM.

Everything is seeded. On different hardware or library versions, the third decimal may occasionally differ.

## Notebook layout

| Section | Report |
|---|---|
| 1–4. Data, models, training | §2–3 |
| 5. Cross-validation | Table 1 |
| 6. Concept probing | Figure 2 |
| 7. SHAP, worked example, choice of block 21 | Figures 3–5 |
| 8. Q1: where do the two models look? | Figure 6 |
| 9. Q2: evidence outside the concepts | Figure 7 |

Figure 1 is the model diagram and has no code.

## Data and models

- Fitzpatrick17k: Groh et al., 2021, via Kaggle (`asosenge/fitzpatrick17k`).
- SkinCon: Daneshjou et al., 2022.
- DINOv3: Siméoni et al., 2025, weights through timm (`vit_large_patch16_dinov3.lvd1689m`), under the DINOv3 license.
