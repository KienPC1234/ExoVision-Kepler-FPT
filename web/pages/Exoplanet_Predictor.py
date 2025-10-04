import streamlit as st
import pandas as pd
import numpy as np
from web.helper import streamlit_custom_components as scc

st.set_page_config(page_title="Exoplanet Predictor", layout="wide")

# ----------------------------
# TITLE + Markdown hướng dẫn
# ----------------------------
st.markdown(
    """
# 🌌 Exoplanet Predictor

Predict exoplanet characteristics either by **manual input** or by **uploading CSV** from K2, KOI, or TESS catalogs.

Use the **Simple** tab for a quick planet check or **Advanced** tab for batch processing.

---
"""
)

with st.expander("ℹ️ How to obtain K2, KOI, or TESS data"):
    st.markdown("""
- **KOI:** [Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu)  
- **K2 mission:** [STScI K2 Archive](https://archive.stsci.edu/k2)  
- **TESS:** [TESS TOI Tables](https://exoplanetarchive.ipac.caltech.edu/docs/tessmission.html)  

Required columns: `pl_radj`, `pl_orbper`, `pl_trandur`, `depth`, `st_teff`, `st_logg`, `st_rad`, `koi_kepmag`  
Optional columns: `koi_impact`, `pl_insol`, `pl_eqt`, `st_dist`
"""
    )

# ----------------------------
# Tabs
# ----------------------------
tab = st.tabs(["Simple", "Advanced"])

# ----------------------------
# TAB 1: Simple
# ----------------------------
with tab[0]:
    st.header("Simple Input - Enter Single Planet Data")

    REQUIRED_FIELDS = [
        ('koi_kepmag', 'Kepler-band brightness (mag)'),
        ('pl_radj', 'Planet radius (R_J)'),
        ('pl_orbper', 'Orbital period (days)'),
        ('pl_trandur', 'Transit duration (hours)'),
        ('depth', 'Transit depth (fraction)'),
        ('st_teff', 'Stellar effective temperature (K)'),
        ('st_logg', 'Stellar surface gravity (dex)'),
        ('st_rad', 'Stellar radius (R_Sun)'),
    ]

    OPTIONAL_FIELDS = [
        ('koi_impact', 'Impact parameter (dimensionless)'),
        ('pl_insol', 'Insolation flux (F_Earth)'),
        ('pl_eqt', 'Planet equilibrium temperature (K)'),
        ('st_dist', 'Stellar distance (pc)')
    ]

    manual_data = {}
    cols = st.columns(2)
    for i, (field, label) in enumerate(REQUIRED_FIELDS):
        col = cols[i % 2]
        manual_data[field] = col.number_input(
            f"{label} *", 
            value=0.0, 
            step=0.01, 
            format="%.5f",
            help=f"{label} (required)"
        )
    for i, (field, label) in enumerate(OPTIONAL_FIELDS):
        col = cols[i % 2]
        manual_data[field] = col.number_input(
            f"{label}", 
            value=np.nan, 
            step=0.01, 
            format="%.5f",
            help=f"{label} (optional)"
        )

    if st.button("Check Planet"):
        df_manual = pd.DataFrame([manual_data]).fillna(0)
        # Derived fields
        df_manual['density_proxy'] = 1 / df_manual['pl_radj'].replace(0, np.nan)**3
        df_manual['habitability_proxy'] = df_manual['pl_orbper'] * 0.7 / df_manual['st_teff'].replace(0, np.nan)
        df_manual['transit_shape_proxy'] = df_manual['depth'] / df_manual['pl_trandur'].replace(0, np.nan)

        # Planet existence check
        if df_manual['pl_radj'].values[0] > 0 and df_manual['pl_orbper'].values[0] > 0:
            st.success("✅ This planet exists")
        else:
            st.error("❌ This planet does not exist")

        st.subheader("Planet Preview")
        scc.embed_iframe("http://103.252.0.76/iframe/plant_preview/", height=600)


# ----------------------------
# TAB 2: Advanced
# ----------------------------
with tab[1]:
    st.header("Advanced Input - Upload CSV (K2 / TESS / KOI)")
    
    csv_type = st.selectbox("Select Data type", ["k2", "tess", "koi"])
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file).fillna(0)

        required_cols = ['pl_radj', 'pl_orbper', 'pl_trandur', 'depth', 'st_teff', 'st_logg', 'st_rad', 'koi_kepmag']
        missing_cols = [c for c in required_cols if c not in df_csv.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            df_csv['density_proxy'] = 1 / df_csv['pl_radj'].replace(0, np.nan)**3
            df_csv['habitability_proxy'] = df_csv['pl_orbper'] * 0.7 / df_csv['st_teff'].replace(0, np.nan)
            df_csv['transit_shape_proxy'] = df_csv['depth'] / df_csv['pl_trandur'].replace(0, np.nan)

            st.subheader("Processed Data Preview")
            st.dataframe(df_csv.head())

            csv_out = df_csv.to_csv(index=False).encode('utf-8')
            st.download_button("Download Processed CSV", csv_out, "exoplanet_processed.csv", "text/csv")
