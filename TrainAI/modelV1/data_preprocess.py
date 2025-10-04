import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

def drop_noise_features(df):
    cols_to_drop = [c for c in df.columns if
                    "err" in c.lower() or
                    "lim" in c.lower() or
                    "symerr" in c.lower() or
                    c in [
                        "ra","dec","x","y","z","htm20","confidence","sectors",
                        "ra_str","dec_str","rastr","decstr",
                        # IDs / display / possible data-leak columns
                        "koi_delivname","toidisplay","pl_tranmid",
                        # proper motion rarely predictive for disposition
                        "st_pmra","st_pmdec",
                        # created elsewhere but ensure removal to avoid leak
                        "release_year","release_month"
                    ]]
    return df.drop(columns=cols_to_drop, errors="ignore")


def load_or_download(url, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        try:
            df = pd.read_csv(url, low_memory=False)
            df.to_csv(filename, index=False)
            print(f"Downloaded {filename} from {url}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}. Please download manually and place in {filename}")
    else:
        df = pd.read_csv(filename, low_memory=False)
        print(f"Loaded local {filename}")
    return df

def standardize_koi(df):
    drop_cols = [
        'rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_vet_stat', 'koi_vet_date', 'koi_pdisposition',
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_disp_prov', 'koi_comment',
        'koi_time0bk', 'koi_time0', 'koi_eccen', 'koi_longp', 'koi_ingress', 'koi_ror', 'koi_srho', 'koi_fittype',
        'koi_sma', 'koi_incl', 'koi_dor', 'koi_limbdark_mod', 'koi_ldm_coeff4',
        'koi_ldm_coeff3', 'koi_ldm_coeff2', 'koi_ldm_coeff1', 'koi_parm_prov', 'koi_max_sngle_ev',
        'koi_max_mult_ev', 'koi_model_snr', 'koi_count', 'koi_num_transits', 'koi_tce_plnt_num',
        'koi_tce_delivname', 'koi_quarters', 'koi_bin_oedp_sig', 'koi_trans_mod', 'koi_model_dof',
        'koi_model_chisq', 'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_smet', 'koi_smass', 'koi_sage',
        'koi_sparprov', 'koi_gmag', 'koi_rmag', 'koi_imag', 'koi_zmag', 'koi_jmag', 'koi_hmag', 'koi_kmag',
        'koi_fwm_stat_sig', 'koi_fwm_sra', 'koi_fwm_sdec', 'koi_fwm_srao', 'koi_fwm_sdeco', 'koi_fwm_prao',
        'koi_fwm_pdeco', 'koi_dicco_mra', 'koi_dicco_mdec', 'koi_dicco_msky', 'koi_dikco_mra', 'koi_dikco_mdec',
        'koi_dikco_msky'
    ]
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1, errors='ignore')
    
    mapping = {
        'koi_period': 'pl_orbper',
        'koi_depth': 'depth',
        'koi_prad': 'pl_radj',
        'koi_steff': 'st_teff',
        'koi_slogg': 'st_logg',
        'koi_srad': 'st_rad',
        'koi_disposition': 'disposition',
        'koi_score': 'confidence',
        'koi_duration': 'pl_trandur',
        'koi_kepmag': 'koi_kepmag',
        'koi_impact': 'koi_impact',
        'koi_insol': 'pl_insol',
        'koi_teq': 'pl_eqt'
    }
    df = df.rename(columns=mapping)
    
    # Unit conversions for KOI
    if 'pl_radj' in df.columns:
        df['pl_radj'] = df['pl_radj'] / 11.209  # Earth radii to Jupiter radii
    if 'depth' in df.columns:
        df['depth'] = df['depth'] / 1e6  # ppm to fraction
    
    # Add st_dist as NaN for KOI (no direct column)
    df['st_dist'] = np.nan
    
    if 'disposition' in df.columns:
        df['disposition'] = df['disposition'].map({'FALSE POSITIVE': 0, 'CANDIDATE': 1, 'CONFIRMED': 1}).fillna(1).astype(int)
    
    return df

def standardize_k2(df):
    drop_cols = [
        'rowid', 'pl_name', 'hostname', 'pl_letter', 'k2_name', 'epic_hostname', 'epic_candname', 'hd_name', 'hip_name',
        'tic_id', 'gaia_id', 'default_flag', 'disp_refname', 'sy_snum', 'sy_pnum', 'sy_mnum', 'cb_flag', 'disc_year',
        'disc_refname', 'disc_pubdate', 'disc_locale', 'disc_facility', 'disc_telescope', 'disc_instrument', 'rv_flag',
        'pul_flag', 'ptv_flag', 'tran_flag', 'ast_flag', 'obm_flag', 'micro_flag', 'etv_flag', 'ima_flag', 'dkin_flag',
        'soltype', 'pl_controv_flag', 'pl_refname', 'pl_orbsmax', 'pl_rade', 'pl_masse', 'pl_massj', 'pl_msinie',
        'pl_msinij', 'pl_cmasse', 'pl_cmassj', 'pl_bmasse', 'pl_bmassj', 'pl_bmassprov', 'pl_dens', 'pl_orbeccen',
        'pl_orbincl', 'pl_tranmid', 'pl_tsystemref', 'ttv_flag', 'pl_ratdor', 'pl_ratror', 'pl_occdep', 'pl_orbtper', 'pl_orblper', 'pl_rvamp', 'pl_projobliq',
        'pl_trueobliq', 'st_refname', 'st_spectype', 'st_met', 'st_metratio', 'st_lum', 'st_age', 'st_vsin',
        'st_rotp', 'st_radv', 'sy_refname', 'rastr', 'decstr', 'glat', 'glon', 'elat', 'elon', 'sy_pm', 'sy_pmra',
        'sy_pmdec', 'sy_plx', 'sy_bmag', 'sy_vmag', 'sy_jmag', 'sy_hmag', 'sy_umag', 'sy_gmag',
        'sy_rmag', 'sy_imag', 'sy_zmag', 'sy_w1mag', 'sy_w2mag', 'sy_w3mag', 'sy_w4mag', 'sy_gaiamag', 'sy_icmag',
        'sy_tmag', 'rowupdate', 'pl_pubdate', 'releasedate', 'pl_nnotes', 'k2_campaigns', 'k2_campaigns_num',
        'st_nphot', 'st_nrvc', 'st_nspec', 'pl_nespec', 'pl_ntranspec', 'pl_ndispec'
    ]
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1, errors='ignore')
    
    # Drop all *_str columns to avoid high cardinality
    str_cols = [col for col in df.columns if col.endswith('str')]
    df = df.drop(str_cols, axis=1, errors='ignore')
    
    mapping = {
        'pl_orbper': 'pl_orbper',
        'pl_trandep': 'depth',
        'pl_radj': 'pl_radj',
        'st_teff': 'st_teff',
        'st_logg': 'st_logg',
        'st_rad': 'st_rad',
        'disposition': 'disposition',
        'discoverymethod': 'discoverymethod',
        'pl_trandur': 'pl_trandur',
        'sy_kepmag': 'koi_kepmag',
        'pl_imppar': 'koi_impact',
        'pl_insol': 'pl_insol',
        'pl_eqt': 'pl_eqt',
        'sy_dist': 'st_dist'
    }
    df = df.rename(columns=mapping)
    
    # Unit conversions for K2
    if 'depth' in df.columns:
        df['depth'] = df['depth'] / 100  # percent to fraction
    
    if 'disposition' in df.columns:
        df['disposition'] = df['disposition'].map({'FALSE POSITIVE': 0, 'REFUTED': 0, 'CANDIDATE': 1, 'CONFIRMED': 1}).fillna(1).astype(int)
    
    return df

def standardize_tess(df):
    mapping = {
        'pl_orbper': 'pl_orbper',
        'pl_trandep': 'depth',
        'pl_rade': 'pl_radj',
        'st_teff': 'st_teff',
        'st_logg': 'st_logg',
        'st_rad': 'st_rad',
        'tfopwg_disp': 'disposition',
        'st_tmag': 'koi_kepmag',
        'pl_trandurh': 'pl_trandur',
        'pl_insol': 'pl_insol',
        'pl_eqt': 'pl_eqt',
        'st_dist': 'st_dist'
    }
    df = df.rename(columns=mapping)
    
    # Unit conversions for TESS
    if 'pl_radj' in df.columns:
        df['pl_radj'] = df['pl_radj'] / 11.209  # Earth radii to Jupiter radii
    if 'depth' in df.columns:
        df['depth'] = df['depth'] / 1e6  # ppm to fraction
    
    # Add koi_impact as NaN for TESS (no direct column)
    df['koi_impact'] = np.nan
    
    if 'disposition' in df.columns:
        df['disposition'] = df['disposition'].map({'FP': 0, 'CP': 1, 'KP': 1, 'PC': 1}).fillna(1).astype(int)
    
    drop_cols = ['rowid', 'toi', 'toipfx', 'tid', 'ctoi_alias', 'pl_pnum', 'rastr', 'decstr', 'toi_created', 'rowupdate']
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1, errors='ignore')
    
    return df

def preprocess_step(df):
    y = df['disposition']
    X = df.drop('disposition', axis=1).copy()
    
    # Convert possible date
    if 'release_date' in X.columns:
        X = X.drop(columns=['release_date'])
    
    # Handle mixed types
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = []
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in object_cols:
        try:
            X[col] = pd.to_numeric(X[col], errors='raise')
            numerical_cols.append(col)
        except ValueError:
            categorical_cols.append(col)
    
    # Remove columns with almost-all NaN
    thresh = 0.3 * len(X)  # keep cols with at least 30% non-null
    drop_high_nan = [c for c in X.columns if X[c].notna().sum() < thresh]
    X = X.drop(columns=drop_high_nan, errors='ignore')
    # re-evaluate lists
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c in X.columns and X[c].notna().any()]
    
    # Impute numerical first
    num_imputer = None
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='median')
        imputed_num = num_imputer.fit_transform(X[numerical_cols])
        X[numerical_cols] = pd.DataFrame(imputed_num, columns=numerical_cols, index=X.index)
    
    # --- CREATE DERIVED FEATURES HERE (AFTER IMPUTE, BEFORE SCALE) ---
    if 'pl_radj' in X.columns:
        # assume pl_radj currently in physical units (not scaled)
        X['density_proxy'] = 1.0 / (X['pl_radj'].replace(0, np.nan) ** 3 + 1e-12)
        X['density_proxy'] = X['density_proxy'].fillna(0.0)
    if 'pl_orbper' in X.columns and 'st_teff' in X.columns:
        X['habitability_proxy'] = (X['pl_orbper'] * 0.7) / (X['st_teff'].replace(0, np.nan) + 1e-12)
        X['habitability_proxy'] = X['habitability_proxy'].fillna(0.0)
    if 'depth' in X.columns and 'pl_trandur' in X.columns:
        X['transit_shape_proxy'] = X['depth'] / (X['pl_trandur'].replace(0, np.nan) + 1e-12)
        X['transit_shape_proxy'] = X['transit_shape_proxy'].fillna(0.0)
    
    # Impute and encode categorical (LabelEncoder per col) — keep mapping
    label_encoders = {}
    cat_imputer = None
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Now scale numerical (after derived features added)
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    if numerical_cols:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Final NaN guard
    if X.isnull().any().any():
        X = X.fillna(0)
    
    # Return label_encoders dict as encoder, for saving later
    return X, y, scaler, label_encoders, num_imputer, cat_imputer

def process_all_csvs(output_csv='data/merged_processed.csv'):
    os.makedirs('data', exist_ok=True)
    os.makedirs('models/v1', exist_ok=True)
    
    # Define URLs and local files
    koi_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
    koi_file = 'dataset/koi_cumulative.csv'
    df_koi = load_or_download(koi_url, koi_file)
    df_koi = drop_noise_features(standardize_koi(df_koi))
    
    k2_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2pandc&format=csv"
    k2_file = 'dataset/k2_pandc.csv'
    df_k2 = load_or_download(k2_url, k2_file)
    df_k2 = drop_noise_features(standardize_k2(df_k2))
    
    tess_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv"
    tess_file = 'dataset/toi.csv'
    df_tess = load_or_download(tess_url, tess_file)
    df_tess = drop_noise_features(standardize_tess(df_tess))
    
    # Merge all standardized dataframes
    dfs = [df_koi, df_k2, df_tess]
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Merged shape: {merged_df.shape}")
    
    merged_df = merged_df.dropna(subset=['disposition'])
    
    # Preprocess the merged dataframe
    X, y, scaler, label_encoders, num_imputer, cat_imputer = preprocess_step(merged_df)
    
    # Save the processed (non-resampled) data
    processed_df = pd.concat([X, y], axis=1)
    
    processed_df.to_csv(output_csv, index=False)
    print(f"Merged processed CSV saved: {output_csv} (shape: {processed_df.shape}, features: {len(X.columns)})")
    merged_df.to_csv("raw.csv", index=False)
    # Save preprocessors
    joblib.dump(scaler, 'models/v1/global_scaler.pkl')
    joblib.dump(num_imputer, 'models/v1/num_imputer.pkl')
    #joblib.dump(cat_imputer, 'models/v1/cat_imputer.pkl')
    joblib.dump(label_encoders, 'models/v1/label_encoders.pkl')

    # Save feature order for inference
    feature_list = X.columns.tolist()
    print(feature_list)
    joblib.dump(feature_list, 'models/v1/feature_list.pkl')
    
    print("Done! The dataset now uses the latest data from NASA Exoplanet Archive and processes consistently.")

if __name__ == "__main__":
    process_all_csvs('data/merged_processed.csv')