
# ISROGENZ_COMPLETE_JUPYTER.py
# Run this file directly in Jupyter or VS Code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks

# =====================
# SOLEXS
# =====================

solexs_file = r"solexs_data\AL1_SLX_L1_20260616_v1.0\SDD2\AL1_SOLEXS_20260616_SDD2_L1.lc.gz"

print("Opening SoLEXS...")
hdul = fits.open(solexs_file)

data = hdul[1].data

time = data["TIME"]
counts = data["COUNTS"]

counts_np = np.array(counts, dtype=np.float64)
counts_clean = np.nan_to_num(counts_np, nan=0)

print("SoLEXS Samples:", len(counts_clean))
print("SoLEXS Max:", np.max(counts_clean))
print("SoLEXS Mean:", np.mean(counts_clean))

plt.figure(figsize=(14,5))
plt.plot(counts_clean[:5000])
plt.title("SoLEXS Light Curve")
plt.grid(True)
plt.show()

peaks, props = find_peaks(
    counts_clean,
    height=30,
    prominence=8,
    distance=300
)

print("SoLEXS Peaks:", len(peaks))

solexs_catalog = pd.DataFrame({
    "Peak_Index": peaks.astype(np.int64),
    "Peak_Time": np.array(time[peaks], dtype=np.float64),
    "Peak_Counts": np.array(counts_clean[peaks], dtype=np.float64)
})

solexs_catalog = solexs_catalog.sort_values(
    by="Peak_Counts",
    ascending=False
)

solexs_catalog.to_csv("solexs_flare_catalog.csv", index=False)

# =====================
# HEL1OS
# =====================

hel_file = r"HLS_20260616_115959_43192sec_lev1_V111\2026\06\16\HLS_20260616_115959_43192sec_lev1_V111\cdte\lightcurve_cdte1.fits"

print("Opening HEL1OS...")
hdul_h = fits.open(hel_file)

data_h = hdul_h[5].data

time_h = data_h["MJD"]
rate_h = data_h["CTR"]

rate_np = np.array(rate_h, dtype=np.float64)

print("HEL1OS Samples:", len(rate_np))
print("HEL1OS Max:", np.max(rate_np))
print("HEL1OS Mean:", np.mean(rate_np))

plt.figure(figsize=(14,5))
plt.plot(rate_np[:5000])
plt.title("HEL1OS Light Curve")
plt.grid(True)
plt.show()

peaks_h, props = find_peaks(
    rate_np,
    prominence=8,
    height=8,
    distance=500
)

print("HEL1OS Peaks:", len(peaks_h))

hel_catalog = pd.DataFrame({
    "Peak_Index": peaks_h.astype(np.int64),
    "MJD": np.array(time_h[peaks_h], dtype=np.float64),
    "CTR": np.array(rate_np[peaks_h], dtype=np.float64)
})

hel_catalog = hel_catalog.sort_values(
    by="CTR",
    ascending=False
)

hel_catalog.to_csv("hel1os_flare_catalog.csv", index=False)

# =====================
# MASTER CATALOG
# =====================

master_catalog = pd.concat([
    solexs_catalog.assign(Source="SoLEXS"),
    hel_catalog.assign(Source="HEL1OS")
], ignore_index=True)

master_catalog.to_csv("master_catalog.csv", index=False)

print("\\nDone!")
print("Generated:")
print("- solexs_flare_catalog.csv")
print("- hel1os_flare_catalog.csv")
print("- master_catalog.csv")
