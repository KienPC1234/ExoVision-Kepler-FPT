![Logo](https://assets.spaceappschallenge.org/media/images/assetstask_01k6td23acefa8qy6s4sy.width-1024.png)

# Exoplanet Detection and Exploration Platform

## Project Overview

Our project is a **web-based platform** for **exoplanet detection and exploration**, powered by **NASA datasets** (Kepler, K2, TESS, KOI). It enables:

1. **Prediction of planetary candidates**

   * Input via tabular metadata (orbital period, radius, stellar parameters, etc.)
   * Upload of lightcurve data (flux time-series)

2. **3D Visualization of planetary systems**

   * Interactive orbital simulations built with **Three.js**


## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/KienPC1234/ExoVision-Kepler-FPT.git
cd ExoVision-Kepler-FPT
```

### 🧪 3. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # For Linux/macOS
# Or use .venv\Scripts\activate on Windows
```

> Once activated, your terminal prompt should show `(.venv)` indicating you're inside the virtual environment.

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

### 5. Run Web App

```bash
streamlit run streamlit_app.py
```
---
### How It Works

* **Tabular Metadata** → Ensemble Stacking (LightGBM, Random Forest, XGBoost, Neural Network) with Logistic Regression meta-learner.
* **Lightcurves** → Transformer model (**PatchTST**) with attention on patched windows.
* **Preprocessing Pipeline** includes dataset merging, cleaning, unit conversions, label encoding, feature engineering, imputation, scaling, balancing (SMOTE), and artifact saving for consistent inference.

Predictions are logged and displayed in both tabular and 3D orbital views. Users can also **train custom models** with built-in tutorials.

---

### Project Structure
```
ExoVision/
├── .streamlit/               # Streamlit configuration files
│   └── config.toml
├── LICENSE                   # Project license information
├── ModelTrainer/             # Core training and model loading logic
│   ├── checkgpu.py           # GPU availability checker
│   ├── modelV1/              # Version 1 of the model pipeline
│   │   ├── data_preprocess.py
│   │   ├── model_builder.py
│   │   ├── model_loader.py
│   ├── modelV2/              # Version 2 of the model pipeline
│   │   ├── check_dataset.py
│   │   ├── data_preprocess.py
│   │   ├── model_builder.py
│   │   ├── model_loader.py
│   └── readme.md             # Internal documentation for ModelTrainer
├── data/                     # Preprocessed and raw data files
│   ├── koi_lightcurves.parquet
│   ├── merged_processed.csv
├── dataset/                  # External datasets used for training and prediction
│   ├── k2_pandc.csv
│   ├── koi_cumulative.csv
│   ├── toi.csv
├── models/                   # Saved models and evaluation artifacts
│   ├── v1/                   # Artifacts from model version 1
│   │   ├── feature_list.pkl, stacking_model.pkl, etc.
│   ├── v2/                   # Artifacts from model version 2
│   │   ├── best_patchtst.pth, y_test.npy
├── readme.md                 # Main project documentation
├── requirements.txt          # Python dependencies
├── static/                   # Static assets for Streamlit (fonts, videos, etc.)
│   ├── *.ttf, *.mp4
├── streamlit_app.py          # Entry point for the Streamlit web application
├── supervisor_config/        # Supervisor configuration files for deployment
│   ├── iframe_loader.conf, streamlit.conf
├── usgi_service.py           # uWSGI service integration script
├── web/                      # Backend and frontend logic for web integration
│   ├── db/                   # Database models and wrappers
│   ├── helper/               # Custom Streamlit components and helpers
│   ├── iframe_loader/        # HTML/JS assets for iframe rendering
│   ├── pages/                # Streamlit page modules (e.g., home, login, prediction)
│   ├── utils/                # Utility functions (auth, routing, etc.)
```
---

## Benefits

* **High Recall & Accuracy** (~90% for metadata, >85% for lightcurves) → fewer missed candidates.
* **User-Friendly**: Simple UI for predictions, uploads, search/export history.
* **Interactive**: Realistic 3D orbital viewer for deeper insights.
* **Efficient & Secure**: Async worker queue, GPU acceleration, authentication, HTTPS.
* **Extensible**: Support for custom model training and experimentation.

---

## Intended Impact

The platform **democratizes exoplanet research** by lowering barriers to entry:

* Researchers → rapid prototyping and exploration.
* Educators → classroom-friendly simulations.
* Students & citizen scientists → accessible tools for discovery.

Our **focus on high recall** addresses the "needle-in-a-haystack" challenge, ensuring rare planetary candidates are not overlooked.

---

## Tech Stack

* **Languages**: Python (ML backend), JavaScript (Three.js frontend)
* **Frameworks**: Streamlit (UI), PyTorch (PatchTST), scikit-learn, LightGBM, XGBoost
* **Data Tools**: Pandas, NumPy, PyArrow, Astropy
* **Deployment**: Docker, Celery/RabbitMQ, Nginx, GPU VPS
* **Other Tools**: Git, Jupyter, Pickle/Joblib

---

## Creativity & Innovation

* Combines **ensemble tree models** (tabular) with **transformer models** (time-series) in one platform.
* **Real-time 3D visualization** brings planetary systems to life.
* **User-oriented workflows** (prediction history, custom training) make research collaborative.
* **PatchTST patching** handles noisy flux data with minimal feature engineering.

---

## Design Considerations

* **Scalability**: Efficient GPU-based processing for billions of flux points.
* **Usability**: Streamlit UI for non-experts, tutorials for new users.
* **Ethics & Security**: Bias mitigation (SMOTE), authentication, privacy compliance.
* **Performance Trade-off**: Prioritized recall over precision.
* **Team Collaboration**: Modular code for parallel development.

---

## Future Directions

* Real-time **TESS data streaming** for live candidate detection.
* Expanded citizen science features for **open participation**.
* Multi-class classification expansion with more nuanced planetary states.

---

🚀 *Making exoplanet research accessible, interactive, and collaborative for everyone.*
