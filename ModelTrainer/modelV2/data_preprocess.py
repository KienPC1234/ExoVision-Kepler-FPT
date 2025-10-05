
"""
- Read dataset/koi_cumulative.csv
- Load lightcurves for each kepid (Kepler) using lightkurve
- Stitch all quarters, normalize 'time' column
- Prioritize flux column: pdcsap_flux > sap_flux > flux
- Write results (time, flux, kepid, label) to dataset/koi_lightcurves.parquet
- Encode labels:
    FALSE POSITIVE -> 0
    CANDIDATE     -> 1
    CONFIRMED     -> 2
"""

import os
import time
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
from lightkurve import search_lightcurve
from fastparquet import write

# ---- config ----
KOI_FILE = "dataset/koi_cumulative.csv"
OUTPUT_FILE = "data/koi_lightcurves.parquet"
PAUSE_BETWEEN = 0.1
MAX_KOI = None      # None = all
MAX_RETRY = 5

# ---- logging setup ----
os.makedirs("logs", exist_ok=True)
log_file = "logs/koi_lightcurves.log"
max_bytes = 50 * 1024 * 1024  # ~50MB
backup_count = 3  
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---- helper ----
def find_flux_column(df):
    for candidate in ["pdcsap_flux", "sap_flux", "flux"]:
        if candidate in df.columns:
            return candidate
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() != 'time']
    return numeric_cols[0] if numeric_cols else None

def safe_download_lightcurves(kepid, max_retry=MAX_RETRY):
    query = f"KIC {kepid}"
    attempt = 0
    while attempt < max_retry:
        try:
            search_res = search_lightcurve(query, mission="Kepler")
            if search_res is None:
                search_res = search_lightcurve(kepid, mission="Kepler")
            if search_res is None:
                return None

            lc_collection = search_res.download_all()
            if lc_collection is None or len(lc_collection) == 0:
                return None
            return lc_collection
        except Exception as e:
            msg = str(e)
            logger.warning(f"Lỗi khi tải {kepid} (lần {attempt+1}/{max_retry}): {msg}")
            if "Not recognized as a supported data product" in msg or msg.endswith(".fits"):
                try:
                    cache_root = os.path.expanduser("~/.lightkurve/cache/mastDownload/Kepler")
                    for root, _, files in os.walk(cache_root):
                        for f in files:
                            if f"{kepid}" in f and f.endswith(".fits"):
                                badfile = os.path.join(root, f)
                                logger.info(f"Xoá file hỏng: {badfile}")
                                os.remove(badfile)
                except Exception as ee:
                    logger.warning(f"Không thể xoá file hỏng: {ee}")
            attempt += 1
            time.sleep(1.0)
    return None

def encode_label(label_str: str) -> int:
    val = label_str.upper().strip()
    if val == "FALSE POSITIVE":
        return 0
    elif val == "CANDIDATE":
        return 1
    elif val == "CONFIRMED":
        return 2
    return -1 
def append_parquet(df: pd.DataFrame, path: str):
    """Append df vào file parquet bằng fastparquet."""
    if not os.path.exists(path):
        write(path, df, compression="gzip")
    else:
        write(path, df, append=True)

# ---- main ----
def main():
    if not os.path.exists(KOI_FILE):
        raise FileNotFoundError(f"Không tìm thấy {KOI_FILE}")

    koi_df = pd.read_csv(KOI_FILE, dtype=str).fillna("")
    n = len(koi_df)
    if MAX_KOI:
        koi_df = koi_df.iloc[:MAX_KOI]

    for idx, row in koi_df.iterrows():
        kepid = str(row.get("kepid", "")).strip()
        label_str = row.get("koi_disposition", "")

        if not kepid:
            logger.warning(f"Bỏ qua dòng {idx} vì không có kepid")
            continue

        label = encode_label(label_str)
        if label == -1:
            logger.warning(f"Nhãn không hợp lệ ({label_str}), bỏ qua {kepid}")
            continue

        logger.info(f"[{idx+1}/{n}] Tải kepid={kepid} label={label_str} (mã={label})")

        try:
            lc_collection = safe_download_lightcurves(kepid)
            if lc_collection is None:
                logger.error(f"Không tải được lightcurves cho {kepid}")
                continue

            combined_lc = lc_collection.stitch() if hasattr(lc_collection, "stitch") else lc_collection
            df = combined_lc.to_pandas()

            if 'time' not in df.columns:
                if df.index is not None and df.index.name is not None:
                    df = df.reset_index()
                else:
                    try:
                        tvals = combined_lc.time.value
                        df = df.reset_index(drop=True)
                        df['time'] = tvals
                    except Exception:
                        raise ValueError("Không có cột 'time' trong DataFrame và không thể lấy từ combined_lc.time")

            flux_col = find_flux_column(df)
            if flux_col is None:
                logger.error(f"Không tìm thấy cột flux cho {kepid}, bỏ qua")
                continue

            df = df[['time', flux_col]].rename(columns={flux_col: 'flux'})
            df = df.dropna(subset=['time', 'flux'])
            if df.empty:
                logger.error(f"Sau drop NaN không còn dữ liệu cho {kepid}")
                continue

            df = df.sort_values('time')
            df['kepid'] = int(kepid)
            df['label'] = label

            # parquet (append)
            append_parquet(df, OUTPUT_FILE)

            logger.info(f"✓ Lấy {len(df)} điểm cho {kepid}, đã ghi vào {OUTPUT_FILE}")

            time.sleep(PAUSE_BETWEEN)

        except Exception as e:
            logger.exception(f"Lỗi khi xử lý {kepid}: {e}")

if __name__ == "__main__":
    main()