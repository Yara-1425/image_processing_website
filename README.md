
# Bayan: Interactive Image Processing Lab 

**Bayan** is an interactive, web-based educational platform designed to help students and computer vision enthusiasts visually explore and experiment with core **Digital Image Processing (DIP)** concepts. By shifting away from dry theoretical formulas, this lab allows users to upload custom images, tweak mathematical parameters dynamically via an intuitive sidebar, and see visual results in real-time.

---

## Developed By
* **Alanoud**
* **Shahad**
* **Lamis**
* **Nada**
* **Yara**

---

##  Core Modules & Features

The platform is meticulously organized into **5 interactive tabs**, mirroring a standard academic image processing curriculum:

### 1. Essentials (Image Fundamentals)
* **Metadata Extraction:** Instantly displays fundamental properties of the uploaded image: width, height, number of color channels, shape matrix, and data type (`dtype`).
* **Color Space Conversion:** Visualizes how an image transitions across different channels:
  * **RGB:** Standard multi-channel additive color model.
  * **Grayscale:** Single-channel intensity mapping essential for feature detection.
  * **HSV (Hue, Saturation, Value):** Decouples color/chroma intensity from lighting/brightness.
* **Dynamic Histogram Visualization:** Renders real-time pixel distribution histograms using `matplotlib` (single curve for Grayscale, and interactive triple R-G-B curves for color images).

### 2. Spatial Domain Enhancement
* **Synthetic Noise Generation:** Implements core degradation models to simulate real-world distortion:
  * **Gaussian Noise:** Emulates electronic circuit noise using normal distribution parameters ($\sigma$).
  * **Salt & Pepper Noise:** Simulates impulse transmission dropouts using random density percentages.
* **Intensity Scaling:** Fine-tunes contrast adjustment ($\alpha$) and brightness bias ($\beta$) through explicit matrix manipulation via `cv2.convertScaleAbs`.
* **Spatial Filters (Denoising):** Provides interactive matrix kernels with selectable window sizes ($3\times3, 5\times5, 7\times7, 9\times9$):
  * *Mean Filter (Blur)*, *Median Filter*, *Gaussian Blur*, *Max Filter*, and *Min Filter*.
* **High-Frequency Enhancement:** Implements a localized **Laplacian Sharpening** operator to amplify edge details.

### 3. Frequency Domain Filters
* **Fast Fourier Transform (FFT):** Converts spatial intensity signals into structural frequency representations using 2D FFT shifts.
* **Interactive Frequency Domain Masking:** Allows users to view and apply distinct frequency boundaries:
  * **Ideal Filters (Low Pass / High Pass):** Sharp, mathematically strict cutoff frequencies.
  * **Butterworth Filters (Low Pass / High Pass):** Smooth, decay-order controlled transitions to reduce ringing artifacts.
* **Spectrum Reconstruction:** Renders the isolated *Filter Mask*, the transformed *Filtered Spectrum*, and the recovered spatial image using Inverse FFT (IFFT).

### 4. Boundary Detection & Segmentation
* **Edge Operators:** Comparative inspection tool using different mathematical spatial derivatives:
  * **Sobel:** Highlights directional gradients along the $X$ and $Y$ axes.
  * **Prewitt:** Computes orthogonal edge magnitude differences using static kernels.
  * **Canny Edge Detector:** Demonstrates multi-stage processing including Gaussian pre-smoothing, gradient tracking, and hysteresis thresholding.
* **Image Binarization (Segmentation):**
  * **Manual Thresholding:** Straightforward global pixel isolation (Binary & Binary Inverted).
  * **Otsu's Thresholding:** Evaluates bimodal histograms to automatically calculate the absolute optimal foreground/background split value.

### 5. Morphological Operations & Object Analysis
* **Structural Kernel Manipulation:** Updates structural matrix geometric shapes (`Square`, `Ellipse`, `Cross`) and sizes dynamically.
* **Basic Morphology Operations:** Visual tracking of geometric alterations using **Erosion** (collapsing boundaries) and **Dilation** (expanding boundaries).
* **Advanced Compounds:** Implements multi-stage **Opening** (Erosion then Dilation to drop background noise) and **Closing** (Dilation then Erosion to patch inner holes).
* **Connected Components Analysis (CCA):** Automatically parses continuous pixel clusters to generate a precise metric count of isolated objects, visualized cleanly via random HSV color-mapping.

---

##  Built With

* [Streamlit](https://streamlit.io/) - The rapid web app framework for Data Science.
* [OpenCV (Open Source Computer Vision Library)](https://opencv.org/) - Advanced real-time image processing operations.
* [NumPy](https://numpy.org/) - High-performance N-dimensional array matrix math.
* [Matplotlib](https://matplotlib.org/) - Static tracking and plotting of color distributions.
* [Pillow (PIL)](https://python-pillow.org/) - Core image loading abstraction layer.

---

## Installation & Setup

Follow these simple steps to run the lab environment locally:

### 1. Clone the Repository
```bash
git clone [https://github.com/Yara-1425/image_processing_website.git](https://github.com/Yara-1425/image_processing_website.git)
cd image_processing_website

```

### 2. Install Dependencies

Ensure you have Python 3.8+ installed. You can install all required libraries at once using the `requirements.txt` file:

```bash
pip install -r requirements.txt

```

### 3. Launch the Application

Run the Streamlit server using the main execution file:

```bash
streamlit run app.py

```
---

## 💡 How to Use the Lab

1. Run the script via terminal to launch the local Streamlit development server.
2. Open the **Control Panel** in the left sidebar.
3. Upload any standard image format (`.jpg`, `.jpeg`, `.png`).
4. Navigate through the **5 tabs** at the top of the interface to experiment, adjust parameters, and visually master image processing!

---

© 2026 Bayan Image Lab | Part of Taibah University Artificial Intelligence Program

