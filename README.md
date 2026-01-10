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



### Training Loop
* Forward Pass: The model predicts 3 EVs, synthesizes virtual exposures, and reconstructs the image.
* Validation: Runs after every epoch to track PSNR, SSIM, and Loss.
* Checkpointing: The "Best Model" is saved whenever validation loss decreases. Routine checkpoints are saved every 5 epochs.
* Logging: Metrics are automatically logged to train_log.csv for analysis.  


## Implementation

The implementation is structured as a modular PyTorch workflow, handling data ingestion, dual-branch modeling, and boosted reconstruction.

### 1. Data Loading

The `ToneMappingDataset` class manages the pairing of Raw/LDR inputs with expert HDR targets.

* **Preprocessing:** Resizes images to `224 × 224` and normalizes pixel values to `[0, 1]`.
* **Histogram Calculation:** Computes a 64-bin intensity histogram on-the-fly for the statistical branch.
* **Tensor Inputs:**
  * **Image:** `[B, 3, 224, 224]` (RGB)
  * **Histogram:** `[B, 64]` (Normalized Vector)
  * **Target:** `[B, 3, 224, 224]` (Ground Truth)



### 2. Model Definition

The core logic is encapsulated in `FullToneMapModelBoosting`:

* **Fusion Module:** Integrates the ResNet spatial encoder and MLP histogram encoder to predict EV brackets.
* **Virtual Camera Layer:** Differentiably applies predicted EVs to the input tensor, creating a stack of 3 virtual exposures (Under, Mid, Over).
* **Reconstruction:** `ReconUNetV2` takes the 9-channel stacked input and produces the final 3-channel HDR output.



### 3. Training Loop

The training is governed by the `train_v2` function:

* **Process:** Iterates through epochs, calculating the composite loss terms (L1, SSIM, VGG, EV Constraints) and updating weights via backpropagation.
* **Validation:** Runs at the end of every epoch to evaluate performance on the validation set.
* **Checkpointing:** Automatically saves the model state (`.pth`) to `v2_checkpoints/` whenever validation loss improves.



### 4. Outputs & Visualization

The repository automatically generates a structured `output/` directory (mapped to `v2_samples/` and `v2_checkpoints/` in the code) containing logs and visual artifacts to monitor training progression.

* **`v2_checkpoints/`**
  * `train_log.csv`: Tracks `train_loss` and `val_loss` per epoch.
  * `model_epochXXX.pth`: Saved model weights.

* **`v2_samples/` (Training Artifacts)**
  * **Reconstruction Triplets:** Visual comparison saved every epoch:  
    `[ Input LDR | Predicted HDR | Ground Truth ]`
  * **Virtual Bracket Previews:** Visualizations of the internally synthesized exposures:  
    `[ Virtual Under | Virtual Mid | Virtual Over ]`  
    *These show exactly what the U-Net "sees" before merging them.*

* **`test_samples/` (Final Evaluation)**
  * Contains predictions on unseen test data to verify generalization capability.


## Results
The model demonstrates strong convergence and generalization, effectively balancing exposure prediction with high-fidelity HDR reconstruction.

### Quantitative Test Metrics
* **Test L1 Loss:** 10461.30 
* **Test MSE:** 1474.03
* **Test PSNR:** 20.31 dB
* **Test SSIM:** 0.6852

### Best Model Performance
* Achieved at **Epoch 20**  
* **Validation Loss:** 0.0709  

### Example Predictions
Ground-truth EV shifts:
```
[-0.27156088  0.65435684  0.43656817 -0.64651066]
```

Predicted 5-stop EV brackets:
```
Sample 1: [-1.5562469  -0.04815363  0.57255673]
Sample 2: [-0.9986939   0.5637927   2.2738926 ]
Sample 3: [-0.9257073   0.19482271  1.0358816 ]
Sample 4: [-0.9391494  -0.15072545  0.43088636]
```

