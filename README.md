# Optimal-Exposure-Bracketing-for-HDR-Capture
A dual-branch deep learning framework that automates exposure selection and HDR synthesis. The pipeline fuses spatial features (via ResNet-18) with global luminance statistics (via Histogram MLP) to regress optimal exposure value (EV) brackets from a single LDR preview. A novel Differentiable Boosting Layer internally synthesizes these predicted exposures, which are then processed by a Reconstruction U-Net to generate high-fidelity, tone-mapped HDR outputs.

## Table of Contents
* [About the Project](#about-the-project)
* [Prerequisites](#prerequisites)
* [Dataset](#dataset)
* [Architecture](#architecture)
* [Training](#training)
* [Implementation](#implementation)
* [Results](#results)
* [Contributing](#contributing)

***
## About the Project
This project introduces a novel, architecture-aware approach to High Dynamic Range (HDR) capture. Unlike traditional methods that rely on heuristic metering, this system uses a Multi-Modal Deep Neural Network to analyze a single Low Dynamic Range (LDR) preview frame.

The model fuses spatial data (RGB image) with statistical data (Luminance Histogram) to intelligently predict an optimal bracket of exposure values (EVs). It then internally synthesizes these exposures using a differential boosting layer and merges them via a U-Net reconstruction head to produce a final, tone-mapped HDR image.

Goal:To automate professional-grade exposure bracketing for mobile and embedded imaging systems, ensuring maximum dynamic range coverage with zero user intervention.

The complete workflow is implemented in a single Jupyter notebook (smg_project_nb.ipynb) designed for GPU acceleration.

## Prerequisites
To run the project, the following Python libraries are required:

* **Python 3.x**
* **PyTorch & Torchvision** (Deep Learning framework)
* **OpenCV (cv2)** (Image processing)
* **NumPy** (Numerical operations)
* **Pandas** (Data handling)
* **Matplotlib** (Visualization)
* **Scikit-learn** (Data splitting and metrics)
* **Pillow (PIL)** (Image loading)
* * **Tqdm (Progress tracking)
 
## Dataset
The project utilizes two complementary datasets to ensure robustness across different scenes and lighting conditions:

### MIT-Adobe FiveK
* Inputs: RAW/low-quality images located in `raw/`
* Targets: Expert-retouched outputs located in `e/`
* Pairing: Matched by filenames

### LVZ-HDR Tone-Mapping Benchmark
* Inputs: Tone-mapped LDR outputs from TMO-Net
* Targets: Original HDR images
* Pairing: Matched by basenames across `.png`, `.jpg`, `.jpeg`

### Pair Construction & EV Labels
* Approximately **2400 Adobe** images and **1600 LVZ** images are sampled
* Dataset split:
  * 80% Train
  * 10% Validation
  * 10% Test
* Paired data exported to:
  * `train_pairs.csv`
  * `val_pairs.csv`
  * `test_pairs.csv`
* Each row stores:
  * `input_path`
  * `target_path`
  * `dataset`
  * `ev_label`

### EV Label Computation
The exposure shift between target and input is computed using:

```
EV ≈ log2( (μ_target + ε) / (μ_input + ε) )
```
Where `μ` represents the mean grayscale intensity and `ε` is a small stability constant.
