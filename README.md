# ☀️ TEAM ISROGENZ

## Solar Flare Detection & Forecasting Platform

ISROGENZ is an AI-powered Solar Flare Detection and Forecasting Platform developed using SoLEXS and HEL1OS datasets. The platform processes FITS files, detects solar flare events, generates flare catalogs, analyzes energy spectra, and forecasts future solar activity using Machine Learning.

## Features

### SoLEXS Analysis

* Read and process FITS light curve files
* Solar flare detection using peak analysis
* Flare catalog generation
* Peak count analysis
* Statistical summaries

### HEL1OS Analysis

* Event file processing
* Energy spectrum visualization
* Event catalog generation
* Channel and energy analysis

### Forecasting

* Machine Learning based flare risk prediction
* Probability estimation
* Risk classification (Low / Medium / High)
* 30-minute prediction horizon

### Data Products

* Flare Catalog Download
* HEL1OS Event Catalog Download
* Master Catalog Generation
* CSV Export Support

## Datasets

### SoLEXS

Solar Low Energy X-ray Spectrometer onboard Aditya-L1.

### HEL1OS

High Energy L1 Orbiting X-ray Spectrometer onboard Aditya-L1.


## Technology Stack

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Astropy
* SciPy
* Scikit-Learn


## Workflow

1. Upload SoLEXS Light Curve FITS File
2. Detect Solar Flares
3. Generate Flare Catalog
4. Upload HEL1OS Event FITS File
5. Generate Energy Spectrum
6. Create Master Catalog
7. Predict Future Solar Activity

## Machine Learning

Current forecasting module uses statistical features extracted from solar X-ray observations:

* Counts
* Mean Counts (10 samples)
* Mean Counts (50 samples)
* Standard Deviation

The model estimates the probability of a future flare event and generates risk classifications.


## Applications

* Space Weather Monitoring
* Satellite Protection
* Communication System Reliability
* Scientific Research
* Solar Activity Analysis

## Developed By

TEAM ISROGENZ

MIT Bengaluru

ISRO Hackathon Project
