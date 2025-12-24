# TACT: Tensor Accelerated Computation of Transcendentals on GPUs

![Status](https://img.shields.io/badge/Status-Under%20Review-yellow)
![Platform](https://img.shields.io/badge/Platform-NVIDIA%20GPUs-green)
![Language](https://img.shields.io/badge/Language-CUDA%20%2F%20C%2B%2B-blue)

**TACT** is a novel software-only framework designed to repurpose idle **Tensor Cores** on modern GPUs for high-performance scientific computing. It addresses the critical resource bottleneck caused by Special Function Units (SFUs) in transcendental function evaluation.

> **⚠️ Notice: Source Code Availability**
>
> The source code and full implementation details are currently **withheld** as this work is under review for publication in **IEEE Transactions on Parallel and Distributed Systems (TPDS)**.
>
> Upon acceptance, the full CUDA implementation and C++ APIs will be made publicly available in this repository.

## 🚀 Key Innovation

Modern GPUs suffer from a resource imbalance: while scalar units (SFUs/SP cores) are saturated by transcendental functions (e.g., `sin`, `cos`, `exp`), the powerful Tensor Cores often remain idle in non-DL workloads.

**TACT bridges this gap by:**
1. **Reformulation:** Mapping Taylor series approximations into **Matrix Multiply-Accumulate (MMA)** operations.
2. **Function Fusion:** Enabling the concurrent computation of multiple distinct functions (e.g., Sine + Cosine + Tangent) in a single kernel launch using a unified matrix operation.
3. **Hardware Agnostic:** Requires no hardware modifications and works on standard NVIDIA GPUs (tested on Turing architecture).

## 📊 Performance Highlights

TACT has been evaluated against standard hardware Special Function Units (SFUs) and Single-Precision (SP) cores. Key results include:

### 1. Multi-Function Throughput
By fusing operations, TACT significantly outperforms dedicated hardware units in multi-function kernels:
* **Speedup:** Up to **3.36x** faster than SFUs.
* **Energy Efficiency:** Up to **85.2%** energy reduction compared to SFUs.

### 2. Real-World Applications
The framework demonstrates substantial gains in common scientific algorithms:
* **Fast Fourier Transform (FFT):**
    * **4.23x Speedup** vs. SFUs.
    * **81.5% Energy Savings** vs. SFUs.
* **Discrete Cosine Transform (DCT):**
    * **1.09x Speedup** and **18.4% Energy Savings**.

### 3. Numerical Accuracy
TACT offers a programmer-defined trade-off between speed and precision. In standard configurations, it maintains **100% numerical accuracy** at a $10^{-5}$ tolerance relative to double-precision CPU baselines (`libm`).

## 🛠 Methodology Overview

TACT shifts the paradigm of function evaluation from scalar sequences to massive parallel matrix processing. The core insight is reformulating polynomial approximations (Taylor/Maclaurin series) so they map perfectly onto the $D = A \times B + C$ datapath of Tensor Cores.

* **Input Packing:** Scalar inputs are intelligently packed into input matrices (`Tile A`) to maximize density.
* **Coefficient Broadcasting:** Polynomial coefficients are loaded into shared memory (`Tile B`) and broadcast across warps to minimize memory traffic.
* **Fused Execution:** A single matrix multiplication computes multiple functions simultaneously by partitioning the coefficient matrix.

## 📫 Contact

For academic inquiries or access regarding peer review, please contact:

**Hedieh Moftakhari**
* M.Sc. in Computer Architecture, Sharif University of Technology
* Email: Hedieh.rm@gmail.com

---
*© 2025 Hedieh Moftakhari. All Rights Reserved.*
