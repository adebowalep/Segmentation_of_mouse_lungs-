# Mouse Lung 3D Segmentation & Geometry Extraction (Light-Sheet Microscopy)
This repo contains a **notebook + helper modules** for segmenting **experimental mouse‑lung light‑sheet volumes** (NRRD) and exporting a **clean 3D segmentation mask** for downstream visualization (e.g., **3D Slicer** / **ITK‑SNAP**) and 3D geometry extraction.


## What this project does

This project builds a practical segmentation pipeline for **3D experimental mouse lung images** acquired with **Light-Sheet microscopy**. The workflow explores multiple classical segmentation approaches (global/local thresholding, multilevel thresholding, clustering, morphology) and converges on a robust solution using **Watershed + K-Means** to produce consistent masks across the full stack of slices.

**Output:** a segmented 3D volume (NRRD) that can be opened in tools such as **3D Slicer** or **ITK-SNAP**, and used to extract/visualize 3D surfaces (e.g., Marching Cubes).

---

## Project motivation

Medical imaging volumes often contain noise, uneven illumination, and complex structures. A single global threshold rarely works across all slices. This work focuses on building a segmentation approach that is:
- **Robust across slices**
- **Reproducible**
- **Exportable to standard medical imaging tools**

---

## Data

- **Format:** NRRD (3D, unlabeled)
- **Example volume size:** ~`[1921, 2861, 858]` (uint16)
- **Spacing:** ~`[2.285, 2.285, 10]` microns (x, y, z)
---

## Methods explored

The notebook documents and compares several approaches:

### 1) Thresholding (baseline)
- Manual thresholding (`np.where`)
- Global Otsu thresholding
- Local/adaptive thresholding
- Mean thresholding
- Multi-Otsu (multi-level) thresholding

### 2) Image enhancement / denoising
- Mean filter
- Median filter
- Gaussian filter

### 3) Clustering + morphology
- Gaussian Mixture Model (GMM) segmentation
- Morphological operations (dilation, erosion, opening, closing)

### 4) Final approach (recommended)
**Watershed segmentation (with marker generation) + K-Means**  
High-level idea:
1. Denoise (Gaussian) and compute edges (Canny)
2. Distance transform to create a “landscape”
3. Detect local maxima → markers/seeds
4. Watershed to obtain regions
5. Use region statistics + **K-Means** to label regions as foreground/background
6. Assemble the final binary mask per slice
7. Stack masks to create a full **3D segmentation volume**

---
## Repository structure 
```text
.
├── main.ipynb
├── requirement.txt
├── src/
├── img/
└── .ipynb_checkpoints/   (optional, auto‑generated)
```

## Setup

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
python -m pip install -U pip
```

### 2) Install dependencies

Use the repo’s dependency file:

```bash
pip install -r requirement.txt
```


## How to run

### Run the notebook

1. Put your NRRD volume somewhere accessible (recommended: `./input/`).
2. Open `main.ipynb` and update the input path if needed.

In the notebook, the default load line looks like:

```python
data, header = read_data("../input/essai03stitching_Subset.nrrd")
```
---

## Exported output (NRRD)

The notebook generates a full 3D segmentation volume by iterating through slices, applying the final method, stacking results, and saving:

- `essai03stitching_Subset_segmentation.nrrd`

## Visualizing results

###  ITK‑SNAP / 3D Slicer (recommended)

1. Open the original `.nrrd` volume
2. Open the segmentation `.nrrd`
3. Render as label map / surface
4. Export mesh if needed (STL/OBJ)


## Acknowledgements

- Experimental context/dataset: **IRCAN (Nice)** and collaborators mentioned in the notebook.
- Tools: NumPy, scikit‑image/OpenCV concepts, scikit‑learn, PyNRRD, ITK‑SNAP, 3D Slicer.
