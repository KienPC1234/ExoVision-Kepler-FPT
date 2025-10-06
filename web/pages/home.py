import streamlit as st

st.set_page_config(page_title="Exoplanet Discovery Platform", layout="wide")
st.markdown("""
# 🌌 Exoplanet Discovery Platform

Welcome to the **Exoplanet Discovery Platform**, a cutting-edge web application designed for astronomers and researchers to **explore, analyze, and discover exoplanets** using data from NASA missions such as **TESS, KOI, and K2**.

---

## 🔭 Explore Exoplanets from NASA Missions
- Analyze planet parameters from **NASA catalogs** in table format.
- Examine **light curves** to determine if an exoplanet exists.
- Detect new exoplanets based on transit signals and other observational data.

## 🚀 Interactive 3D Planet Simulations
- Visualize newly discovered exoplanets in **3D**.
- Understand the planet's size, orbit, and other physical characteristics.
- Explore planetary systems dynamically to gain insights into habitability and structure.

## 🖼️ 3D Showcase
""", unsafe_allow_html=True)

# Thêm iframe 3D showcase
st.components.v1.iframe("https://iframe.fptoj.com/iframe/plant_preview/", height=500)

st.markdown("""
---

## 🧠 Researcher-Friendly and Customizable
- Researchers can **reuse the model architecture** to train on their own datasets.
- Supports **advanced feature analysis** and simulation pipelines.
- Provides detailed **guides and tutorials** to help integrate your own data.

## ⚡ Key Features
- **Single Planet Check:** Input planet parameters to preview and validate existence.
- **Batch Processing:** Upload CSV files from TESS, KOI, or K2 for large-scale analysis.
- **Derived Metrics:** Automatic calculation of density, habitability, and transit shape proxies.
- **Interactive Visuals:** 3D rendering and simulation of planetary systems.
- **Extensible Architecture:** Adapt the model for your own research and experiments.

---

### 🌟 Why Use Our Platform?
This platform bridges the gap between **astronomical data** and **actionable insights**, allowing researchers to quickly identify potential exoplanets, simulate their characteristics, and extend models to new datasets. Whether you're analyzing Kepler data, K2 mission targets, or TESS observations, our tools make exoplanet discovery accessible and intuitive.

---

### 📂 Source Code

To explore the full implementation of our project, please visit the GitHub repository:

🔗 [ExoVision-Kepler-FPT](https://github.com/KienPC1234/ExoVision-Kepler-FPT)

---

> “Explore, analyze, and visualize exoplanets like never before. Bring your data and see what new worlds await!”
""", unsafe_allow_html=True)
