# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from astropy.io import fits, ascii
import torch
from tsai.all import load_learner

# -----------------------------
# Load trained PatchTST model
# -----------------------------
@st.cache_resource
def load_model(path='patchtst_learner.pkl', device='cpu'):
    learner = load_learner(path, cpu=(device=='cpu'))
    learner.dls.device = torch.device(device)
    return learner

device = 'cuda' if torch.cuda.is_available() else 'cpu'
learner = load_model('patchtst_learner.pkl', device=device)

# -----------------------------
# Helper to read flux data
# -----------------------------
def read_flux_file(file):
    fname = Path(file.name)
    ext = fname.suffix.lower()
    df = None

    if ext in ['.fits', '.fit']:
        hdul = fits.open(file)
        for hdu in hdul:
            if 'TIME' in hdu.columns.names and any(f in hdu.columns.names for f in ['SAP_FLUX','PDCSAP_FLUX','FLUX']):
                time_col = 'TIME'
                flux_col = [f for f in ['SAP_FLUX','PDCSAP_FLUX','FLUX'] if f in hdu.columns.names][0]
                df = pd.DataFrame({
                    'time': hdu.data[time_col],
                    'flux': hdu.data[flux_col]
                })
                break
        hdul.close()
    elif ext in ['.tbl', '.txt']:
        df = ascii.read(file).to_pandas()
    elif ext in ['.csv', '.tsv']:
        df = pd.read_csv(file)
    else:
        st.warning(f"Unsupported file type: {ext}")
        return None

    # standardize column names
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if 'time' in df.columns and 'flux' in df.columns:
        if 'kepid' not in df.columns:
            df['kepid'] = -1
        df = df[['kepid', 'time', 'flux']]
        return df
    else:
        st.warning(f"File {fname} missing required columns ['time','flux']")
        return None

# -----------------------------
# Helper to predict
# -----------------------------
def predict_flux(df):
    # df: ['kepid','time','flux']
    # normalize flux if needed
    flux = df['flux'].values.astype(np.float32)
    flux = (flux - np.nanmean(flux)) / np.nanstd(flux)
    # reshape for PatchTST input: (B,L,C)
    seq_len = learner.dls.train.one_batch()[0].shape[1]
    # create windows if flux longer
    X_windows = []
    idx_windows = []
    step = seq_len // 2
    for i in range(0, len(flux)-seq_len+1, step):
        X_windows.append(flux[i:i+seq_len])
        idx_windows.append(list(range(i, i+seq_len)))
    X_windows = np.array(X_windows)
    X_windows = torch.tensor(X_windows).unsqueeze(-1).to(learner.dls.device)
    
    learner.model.eval()
    with torch.no_grad():
        logits = learner.model(X_windows)
        probas = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()  # probability of class 1
        preds = logits.argmax(dim=-1).cpu().numpy()

    # merge idx & probability
    results = []
    for idx, p, pred in zip(idx_windows, probas, preds):
        for j in idx:
            results.append({'idx': j, 'prob_1': p, 'class': int(pred)})
    results_df = pd.DataFrame(results)
    return results_df

# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(page_title="Exoplanet Flux Prediction", layout="wide")
st.title("Exoplanet Flux Prediction using PatchTST")
st.markdown("""
This page allows you to predict exoplanet transit probabilities from flux time series.

**Instructions:**
- Supported file formats: `.fits`, `.tbl`, `.csv`, `.tsv`.
- Required columns: `TIME` and one of `SAP_FLUX`, `PDCSAP_FLUX`, `FLUX` (case insensitive).  
- Optional: `KEPID`. If missing, a default value -1 will be assigned.
- Tab 1: Upload multiple files for a single target. Each file will be processed individually.
- Tab 2: Upload preprocessed file with columns ['kepid','time','flux'] for quick prediction.
- Output: CSV with columns `idx`, `prob_1`, `class` (0/1).
""")

tab1, tab2 = st.tabs(["Upload raw flux files", "Upload preprocessed table"])

# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    st.subheader("Upload raw flux files (FITS/TBL/CSV)")
    files = st.file_uploader("Select files", type=['fits','fit','tbl','csv','tsv'], accept_multiple_files=True)
    kepid_input = st.text_input("Enter KEPLER ID (optional, will use -1 if empty)")
    if st.button("Predict for uploaded files"):
        if files:
            all_results = []
            for f in files:
                df = read_flux_file(f)
                if df is not None:
                    if kepid_input:
                        df['kepid'] = int(kepid_input)
                    res_df = predict_flux(df)
                    res_df['filename'] = f.name
                    all_results.append(res_df)
            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)
                st.dataframe(final_df.head(20))
                csv_bytes = final_df.to_csv(index=False).encode()
                st.download_button("Download results CSV", data=csv_bytes, file_name="prediction_results.csv")
        else:
            st.warning("No files uploaded!")

# -----------------------------
# Tab 2
# -----------------------------
with tab2:
    st.subheader("Upload preprocessed table ['kepid','time','flux']")
    pre_file = st.file_uploader("Select preprocessed CSV/TBL/TXT", type=['csv','tbl','txt'])
    if st.button("Predict preprocessed table"):
        if pre_file:
            df = read_flux_file(pre_file)
            if df is not None:
                res_df = predict_flux(df)
                st.dataframe(res_df.head(20))
                csv_bytes = res_df.to_csv(index=False).encode()
                st.download_button("Download results CSV", data=csv_bytes, file_name="prediction_results_preprocessed.csv")
        else:
            st.warning("No file uploaded!")
