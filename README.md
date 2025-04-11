# Fuzzy Intensification‑Based Low Contrast Image Enhancement & Brightness Preservation

A Python toolkit that implements and compares a suite of fuzzy‑driven and classical histogram equalization algorithms for low‑contrast image enhancement, while preserving brightness. Includes quantitative quality metrics and visualizations to help you choose the best approach for your images.

---

## 🚀 Features

- **Fuzzy Dissimilarity Histogram Equalization**  
  – Computes a fuzzy dissimilarity histogram to drive adaptive contrast enhancement.  
  – `fuzzy1.py`, `fuzzy2.py`, `fuzzy3.py`, `fuzz4.py`

- **Bi‑Histogram Equalization Variants**  
  – Brightness Preserving Bi‑Histogram Equalization (BBHE)  
  – Dual Sub‑Image Histogram Equalization (DSIHE)  
  – Standard Deviation‑Split Bi‑Histogram Equalization  
  – Vertical + Horizontal Bi‑Histogram Equalization  
  – `bhe.py`, `bhe1.py`, `bhe2.py`, `bheclr.py`, `bheclr1.py`, `bhe2clr.py`, `horiverti.py`

- **Classic Histogram Equalization**  
  – Grayscale & color implementations for baseline comparison.  
  – `he.py`, `heclr.py`

- **Objective Quality Metrics**  
  – **Entropy** – measures information content  
  – **PSNR** (Peak Signal‑to‑Noise Ratio)  
  – **AMBE** (Absolute Mean Brightness Error)  
  – **SSIM**, **VSI**, **MMSIM**, **GMSD** – advanced perceptual metrics  
  – All scripts report metrics automatically after enhancement.

- **Visualization & Reporting**  
  – Histograms and CDF plots via Matplotlib  
  – Side‑by‑side before/after display with OpenCV  

---
