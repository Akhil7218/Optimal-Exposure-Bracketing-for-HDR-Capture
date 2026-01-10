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
