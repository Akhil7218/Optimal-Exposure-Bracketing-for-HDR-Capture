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

## Architecture
<img width="1266" height="534" alt="image" src="pipeline.png" />

The system implements an end-to-end **Multi-Modal Deep Neural Network** that fuses spatial and statistical information to predict optimal exposure brackets and reconstruct a high-dynamic-range (HDR) image from a single LDR snapshot.

### Spatial Context Encoder (ResNet-18)
* Backbone: ResNet-18 (truncated, ImageNet pre-trained)  
* Input: Single LDR preview frame (3 × 224 × 224)  
* Extracts high-level semantic and texture features  
* Captures global scene context such as sky regions, shadows, and complex structures  

### Global Illumination Encoder (Histogram Branch)
* Computes a 64-bin grayscale luminance histogram from the input image  
* Processed using a dedicated 3-layer Multi-Layer Perceptron (MLP)  
* Explicitly models global pixel intensity distribution  
* Enables differentiation between high-key and low-key scenes  

### Feature Fusion & EV Regression Head
* Concatenates:
  * 512-dim spatial embedding (from ResNet-18)  
  * 64-dim lighting embedding (from histogram MLP)  
* Fused features are passed through a fully connected EV regression head  
* Predicts an optimal Exposure Value (EV) bracket  
* Output: EV triplet (e.g., [-2.0, 0.0, +2.0])  

### Differentiable Boosting Layer (Virtual Camera)
* Acts as a virtual exposure synthesis module inside the network  
* Applies predicted EVs using the physical exposure formulation:  
  `I_virtual = I_input × 2^EV`  
* Generates synthetic underexposed, normal, and overexposed images  
* Eliminates the need for capturing multiple physical exposures  

### HDR Reconstruction Network (Recon-UNet)
* Input: Channel-wise concatenation of virtual exposures (9 × H × W)  
* Architecture: Modified U-Net with skip connections  
* Merges highlight, mid-tone, and shadow information  
* Outputs a final tone-mapped, artifact-free HDR image  


## Training

The model is trained end-to-end using a **multi-term objective function** designed to jointly optimize **exposure prediction accuracy** and **HDR reconstruction quality**.

### Loss Functions

The network minimizes a weighted composite loss:

```
L_total = L_recon + λ_ssim·L_ssim + λ_vgg·L_perceptual + L_ev_regularization
```

#### Reconstruction (L_recon)
* L1 loss ensures pixel-level fidelity between the reconstructed output and the ground truth HDR target.

#### Structural Similarity Loss (L_ssim)
* Weighted at 0.2, this maximizes the SSIM index to preserve structural details and texture information.

#### Perceptual Loss (L_perceptual)
* Weighted at 0.05, this uses a frozen VGG-16 network to minimize the distance between high-level feature maps, aligning the output with human visual perception.

#### EV Regularization Loss( L_ev_regularization)
A custom triplet of losses ensures the predicted exposure bracket is valid:Center Loss:
* Enforces the "Normal" exposure to match the ground truth EV shift.
* Ordering Loss: Enforces $EV_{under} < EV_{normal} < EV_{over}$.
* Spacing Loss: Penalizes brackets that are too narrow, encouraging a wide dynamic range coverage.

---

### Hyperparameters

* Optimizer: **AdamW** (`weight_decay = 1e-5`)  
* Learning Rate: `1 × 10⁻⁴`  
* Scheduler: **ReduceLROnPlateau**  
  * Factor: `0.5`  
  * Patience: `3`  
* Batch Size: `16`  
* Epochs: `40`  
* Gradient Clipping:  
  * L2 norm capped at `1.0` to prevent exploding gradients  

---

### Training Loop

* **Forward Pass**  
  * Predicts EV triplet  
  * Synthesizes virtual exposures via boosting layer  
  * Reconstructs HDR output using Recon-UNet  

* **Validation**  
  * Executed after every epoch  
  * Tracks PSNR, SSIM, and total loss  

* **Checkpointing**  
  * Best model saved when validation loss improves  
  * Routine checkpoints saved every 5 epochs  

* **Logging**  
  * Training and validation metrics logged to `train_log.csv`  
  * Enables post-training analysis and visualization  

