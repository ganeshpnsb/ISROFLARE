import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
import tempfile
import joblib

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="ISROGENZ",
    page_icon="☀️",
    layout="wide"
)

st.markdown("""
<style>

            .stApp{
background:#0B0F19;
}

[data-testid="stMetric"]{
background:#151A28;
padding:15px;
border-radius:15px;
border:1px solid #2A324A;
}

[data-testid="stDataFrame"]{
background:#151A28;
border-radius:15px;
}

[data-testid="stHeader"]{
background:rgba(0,0,0,0);
}

</style>
""",unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;
font-size:72px;
font-weight:800;
color:white;'>

☀️ ISROGENZ

</h1>

<h3 style='text-align:center;color:white;'>

Solar Flare Intelligence Platform

</h3>
""",unsafe_allow_html=True)

st.info(
"🛰️ Real-Time Solar Activity Monitoring using SoLEXS + HEL1OS + Machine Learning"
)


# ----------------------------
# TABS
# ----------------------------

tab1, tab2, tab3 = st.tabs([
    "Flare Detection",
    "Forecasting",
    "About"
])

# ============================
# TAB 1 - FLARE DETECTION
# ============================

with tab1:

    st.subheader("SoLEXS Data")

    solexs_file = st.file_uploader(
        "Upload SoLEXS Light Curve",
        type=["fits", "gz"],
        key="solexs"
    )

    st.subheader("HEL1OS Data")

    hel1os_file = st.file_uploader(
        "Upload HEL1OS Event File",
        type=["fits"],
        key="hel1os"
    )

    if solexs_file is not None:

        try:

            file_bytes = solexs_file.getvalue()

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            hdul = fits.open(temp_path)

            st.success("FITS File Loaded Successfully")

            data = hdul[1].data
            

            columns = list(data.names)

            # =========================
            # SOLEXS LIGHT CURVE MODE
            # =========================

            if "COUNTS" in columns:

                st.success("☀️ SoLEXS Light Curve Detected")

                counts = np.array(
                    data["COUNTS"],
                    dtype=np.float64
                )

                counts = np.nan_to_num(
                    counts,
                    nan=0
                )

                st.session_state["counts"] = counts

                peaks, _ = find_peaks(
                    counts,
                    height=30,
                    prominence=8,
                    distance=300
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "🌞 Samples",
                        len(counts)
                    )

                with col2:
                    st.metric(
                        "🔥 Peaks",
                        len(peaks)
                    )

                with col3:
                    st.metric(
                        "⚡ Max Counts",
                        int(np.max(counts))
                    )

                with col4:
                    st.metric(
                        "📊 Mean Counts",
                        round(np.mean(counts), 2)
                    )

                fig, ax = plt.subplots(
                    figsize=(12, 5)
                )

                ax.plot(
                    counts[:5000],
                    linewidth=1.5
                )
                visible_peaks = peaks[
                    peaks < 5000
                ]

                ax.scatter(
                    visible_peaks,
                    counts[visible_peaks],
                    color="red"
                )

                ax.set_title(
                    "☀️ Solar Flare Detection Dashboard",
                    fontsize=18,
                    color="white"
                )

                ax.set_xlabel(
                    "Time Samples",
                    color="white"
                )

                ax.set_ylabel(
                    "X-Ray Counts",
                    color="white"
                )

                ax.grid(alpha=0.3)
                ax.set_facecolor("#111827")
                fig.patch.set_facecolor("#111827")
                ax.tick_params(colors="white")

                for spine in ax.spines.values():
                    spine.set_color("white")

                st.pyplot(fig)
                st.markdown("## 📈 Solar Statistics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Highest Peak",
                        round(np.max(counts), 2)
                    )

                with col2:
                    st.metric(
                        "Average Counts",
                        round(np.mean(counts), 2)
                    )

                with col3:
                    st.metric(
                        "Detected Events",
                        len(peaks)
                    )

                solexs_catalog = pd.DataFrame({
                   "Source": "SoLEXS",
                   "Peak_Index": peaks,
                  "Peak_Counts": counts[peaks],
                   "Energy": np.nan,
                   "Channel": np.nan
                })
                
                solexs_catalog["Source"] = "SoLEXS"

                st.session_state["solexs_catalog"] = solexs_catalog

                st.write(
                    "### Flare Catalog"
                )

                st.dataframe(
                    solexs_catalog
                )

                st.download_button(
                    "Download Catalog",
                    solexs_catalog.to_csv(
                        index=False
                    ),
                    "solexs_flare_catalog.csv"
                )
        except Exception as e:
            st.error(f"SoLEXS Error: {e}")

    if hel1os_file is not None:

        try:
            file_bytes = hel1os_file.getvalue()

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            hdul = fits.open(temp_path)
            data = hdul[1].data
            columns = list(data.names)

            if "ener" in columns and "chn" in columns:
                st.success("🛰️ HEL1OS Event File Detected")

                energies = np.array(
                    data["ener"],
                    dtype=np.float64
                )

                energies = np.nan_to_num(
                    energies,
                    nan=0
                )

                st.session_state["energies"] = energies

                hel1os_catalog = pd.DataFrame({
                   "Source": "HEL1OS",
                  "Peak_Index": np.nan,
                  "Peak_Counts": np.nan,
                  "Energy": data["ener"],
                   "Channel": data["chn"]
               })

                st.session_state["hel1os_catalog"] = hel1os_catalog
                if "mjd" in columns:
                    event_catalog = pd.DataFrame({
                        "Event_Index": np.arange(len(data)),
                        "Energy": data["ener"],
                        "Channel": data["chn"],
                        "MJD": data["mjd"]
                    })
                else:
                    event_catalog = pd.DataFrame({
                        "Event_Index": np.arange(len(data)),
                        "Energy": data["ener"],
                        "Channel": data["chn"]
                    })

                st.write("### HEL1OS Event Catalog")
                st.dataframe(event_catalog.head(100))

                st.download_button(
                    "Download HEL1OS Event Catalog",
                    event_catalog.to_csv(index=False),
                    "hel1os_event_catalog.csv"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Events", len(energies))

                with col2:
                    st.metric("Max Energy", round(np.max(energies), 2))

                with col3:
                    st.metric("Mean Energy", round(np.mean(energies), 2))

                fig, ax = plt.subplots(
                    figsize=(12, 5)
                )

                ax.hist(
                    energies,
                    bins=100
                )

                ax.set_title(
                    "HEL1OS Energy Spectrum",
                    color="white"
                )

                ax.set_xlabel(
                    "Energy",
                    color="white"
                )

                ax.set_ylabel(
                    "Counts",
                    color="white"
                )

                ax.tick_params(
                    colors="white"
                )
                for spine in ax.spines.values():
                    spine.set_color("white")

                ax.grid(alpha=0.3)

                ax.set_facecolor("#111827")
                fig.patch.set_facecolor("#111827")

                st.pyplot(fig)

                if "mjd" in columns:
                    spectrum_df = pd.DataFrame({
                        "MJD": data["mjd"],
                        "Energy": data["ener"],
                        "Channel": data["chn"]
                    })
                else:
                    spectrum_df = pd.DataFrame({
                        "Energy": data["ener"],
                        "Channel": data["chn"]
                    })

                csv = spectrum_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download HEL1OS Spectrum Data",
                    data=csv,
                    file_name="hel1os_spectrum.csv",
                    mime="text/csv"
                )

            else:
                st.warning(
                    f"Unsupported HEL1OS FITS format\n\nColumns: {columns}"
                )

        except Exception as e:
            st.error(f"HEL1OS Error: {e}")
       

# ============================
# TAB 2 - FORECASTING
# ============================

with tab2:

    st.header(
        "Solar Flare Forecast"
    )

    try:

        model = joblib.load(
            "solar_flare_forecast_model.pkl"
        )

        st.success(
            "Forecast Model Loaded"
        )

        if "counts" in st.session_state:

            counts = st.session_state[
                "counts"
            ]

            current_count = counts[-1]

            mean_10 = np.mean(
                counts[-10:]
            )

            mean_50 = np.mean(
                counts[-50:]
            )

            std_10 = np.std(
                counts[-10:]
            )

            features = pd.DataFrame({
    "counts": [current_count],
    "mean_10": [mean_10],
    "mean_50": [mean_50],
    "std_10": [std_10]
           })
            st.subheader("Debug Information")

            st.write("Features sent to model:")
            st.dataframe(features)

            st.write("Raw model output:")
            st.write(model.predict_proba(features))

            prediction = model.predict(
                features
            )[0]

            probability = model.predict_proba(
                features
            )[0][1]

            if probability < 0.30:
                risk = "🟢 LOW"
            elif probability < 0.70:
                risk = "🟡 MEDIUM"
            else:
                risk = "🔴 HIGH"

            st.markdown(
                f"# {risk}"
            )

            st.progress(
                float(probability)
            )
            st.metric(
                "Forecast Confidence",
                f"{probability*100:.2f}%"
            )
            st.metric(
                   "Prediction Horizon",
                  "30 Minutes"
            )

            if probability > 0.7:
                st.error("🔴 High Probability Solar Activity")
            elif probability > 0.3:
                st.warning("🟡 Moderate Solar Activity")
            else:
                st.success("🟢 Low Solar Activity")

            st.write(
                f"Flare Probability: {probability:.2%}"
            )

        else:

            st.info(
                "Upload a FITS file in the Flare Detection tab first."
            )

    except Exception as e:

        st.error(
            f"Forecast Error: {e}"
        )

# ============================
# TAB 3 - ABOUT
# ============================

with tab3:
    if (
        "solexs_catalog" in st.session_state
        and
        "hel1os_catalog" in st.session_state
    ):

        master_catalog = pd.concat(
            [
                st.session_state["solexs_catalog"],
                st.session_state["hel1os_catalog"]
            ],
            ignore_index=True
        )

        st.dataframe(master_catalog)
        st.metric(
          "Total Combined Events",
           len(master_catalog)
        )
        st.write(master_catalog["Source"].value_counts())

        st.download_button(
            "Download Master Catalogue",
            master_catalog.to_csv(index=False),
            "master_catalog.csv"
        )
        st.metric(
        "Mission Status",
        "ONLINE"
    )

    st.markdown("""
### Objective

Detect solar flares from SoLEXS observations,
analyze HEL1OS event data,
generate a unified catalogue,
and forecast future flare activity using Machine Learning.
""")


    st.markdown("""
# About ISROGENZ

ISROGENZ is an AI-powered Solar Flare Detection and Forecasting Platform.

### 📡 Data Sources
- SoLEXS
- HEL1OS

### 🔥 Features
- FITS Processing
- Solar Flare Detection
- Peak Analysis
- Machine Learning Forecasting
- Real-Time Visualization

### 🛰️ Applications
- Space Weather Monitoring
- Satellite Safety
- Communication Systems
- Scientific Research

### 👨‍💻 Developed By
ISROGENZ TEAM
MIT Bengaluru
""")

    st.markdown("### 📊 Current Capabilities")

    st.write("✓ SoLEXS Light Curve Analysis")
    st.write("✓ Solar Flare Detection")
    st.write("✓ Flare Catalog Generation")
    st.write("✓ Machine Learning Forecasting")
    st.write("✓ FITS File Processing")

    st.markdown("""
---
<center>

🛰️ ISROGENZ v1.0 | Solar Flare Intelligence Platform

</center>
""", unsafe_allow_html=True)

 