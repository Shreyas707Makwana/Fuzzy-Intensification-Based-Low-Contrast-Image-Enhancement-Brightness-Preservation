# Fuzzy Intensification‑Based Image Enhancement

## Project Overview
This repository provides simple Python scripts to enhance low‑contrast images and preserve brightness using both fuzzy‑intensification and classical histogram equalization methods. Compare multiple algorithms and see quality metrics and visualizations automatically.

## Features
- **Fuzzy Intensification**: Adaptive contrast enhancement driven by fuzzy histograms  
- **Bi‑Histogram Equalization**: Mean‑split, standard‑split, vertical/horizontal variants  
- **Classic Equalization**: Grayscale & color implementations  
- **Quality Metrics**: Entropy, PSNR, AMBE, SSIM, VSI, MMSIM, GMSD  
- **Visualization**: Before/after display and histogram/CDF plots

## Quick Guide
1. **Clone & install**  
   ```bash
   git clone https://github.com/your‑username/DesignProject.git
   cd DesignProject
   pip install numpy opencv-python scipy scikit-image matplotlib
2. **Add your image**
   Place your_image.tiff (or any supported format) in the project folder.   
3. **Run an enhancement script**
   ```bash
   python bhe.py      # Bi‑Histogram Equalization example
   python fuzzy1.py   # Fuzzy intensification example
4. **View Results**
   - OpenCV windows show before/after images
   - Matplotlib plots histograms/CDFs
   - Console prints out all metrics
Happy Enhancing!
Bring your low‑contrast images to life with fuzzy intelligence. 🎨✨
   
