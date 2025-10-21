# AFMimaging - FastRBFInterpolator2D
Approximate GPU-accelerated RBF interpolation for extremely large irregular point clouds — designed for AFM (Atomic Force Microscopy) scanning where sample points are not sampled on a regular grid.

This library converts `(X, Y, Z)` scattered AFM measurements into a smooth `(Nx × Ny)` 2D height/brightness map using **local RBF interpolation + GPU acceleration (Apple Metal / MPS)**.

---

## 🧠 Background: Why this exists

- to learn about data processing in relation to AFM

In AFM, especially when scanning:
- with **nonlinear piezo motion**,
- during **complex force maps**,
- or when using **on-the-fly feedback corrections**,

the actual `(X, Y)` acquisition coordinates are *not uniformly spaced*.  
The resulting topography or property map is **scattered** rather than pixel-aligned.

Traditional interpolation like `scipy.interpolate.griddata()` quickly becomes **too slow** or **memory-heavy** at millions of points.

AFM researchers often want:
| Requirement | Why |
|------------|-----|
| Smooth surface | Represents underlying height/material physics |
| Massive point counts (~10M) | High resolution / oversampling |
| Fast execution | Near real-time visual feedback while scanning |
| GPU acceleration | Especially on Apple M1/M2 (Metal, not CUDA) |

This package provides a practical solution for exactly this problem.

---

## ✨ Key Features

✅ Handles **10 million+ scattered AFM sample points**  
✅ Smooth surface via **Gaussian RBF interpolation**  
✅ Optimized for **Apple Silicon (M1/M2) Metal GPU backend**  
✅ Local interpolation (k-NN) → linear scaling  
✅ Works for **topography, modulus maps, adhesion, stiffness**, etc  
✅ Produces a clean, regular `(Nx × Ny)` AFM image grid

---

## 🔧 Installation

```bash
pip install numpy torch scikit-learn streamlit
````

*(no CUDA required — M1 automatically uses Metal backend)*

---

## 🧱 Usage

### contact point streamlit app
```bash
streamlit run contact_stream.py
````
### RBF interpolator 

```python
from fast_rbf_interp import FastRBFInterpolator2D
import numpy as np

# irregular AFM coordinates (e.g. real scanner positions)
X = ...  # float32 array of shape (N,)
Y = ...  # float32 array of shape (N,)
Z = ...  # signal: height / force / stiffness etc (float32)

interp = FastRBFInterpolator2D(
    grid_size=(1024, 1024),  # output AFM image resolution
    neighbors=64,            # local RBF neighborhood
    epsilon=0.3,             # smoothness
    device='mps'             # Apple M1/M2 GPU
)

Z_img = interp.fit_transform(X, Y, Z)
```

The output is a clean `float32` 1024×1024 AFM image:

* ready for visualization
* usable for segmentation
* compatible with common AFM image processing pipelines

---

## ⚙️ Performance

| Dataset size      | Example                      | Runtime (M1 Pro) |
| ----------------- | ---------------------------- | ---------------- |
| 1 million points  | QNM map or spectroscopy grid | ~4–5 s           |
| 10 million points | high-res oversampled scan    | 30–60 s          |
| 2048×2048 grid    | large stitched AFM image     | supported        |

Unlike full RBF interpolation (`O(N^3)` and impossible at large N),
this method is **local**: `O(N log N)` for neighbor search and `O(grid_size × k)` for interpolation.

---

## AFM-specific notes

### ✅ Works best for:

* Nonlinear scanners without closed-loop XY control
* Oversampled spiral/serpentine scanning trajectories
* Pixel alignment after drift correction
* Force-volume (FV) or fast force mapping
* Mechanical property mapping (adhesion, stiffness, modulus...)

### ✅ Smoothness preserves physics

Gaussian RBF produces a *physically plausible* “continuous surface” — height does not abruptly jump between neighboring values (unlike nearest-neighbor regridding).

---

## Roadmap

* [ ] Batch interpolation for time-series AFM scans
* [ ] Adaptive `epsilon` for nonuniform sampling density
* [ ] FAISS backend for 10× faster neighbor search
* [ ] PyTorch 2.3 compile/export to Metal shader
* [ ] Optional kriging mode for geostatistical AFM

---

## Citation

 .... coming soon 

---

## License

MIT

---

