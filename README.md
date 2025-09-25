# CMAP Scan MUNE Algorithms

This repository contains the Python code and sample data for the Motor Unit Number Estimation (MUNE) algorithms **STEPIX**, **CDIX**, and **Stairfit**. This code is supplementary material for our manuscript "Reliability and Agreement of CMAP Scan–Derived MUNE Algorithms in Healthy Individuals."

The primary tool for using this repository is the included Jupyter Notebook, RunAnalysis.ipynb which provides:
1.  A step-by-step visualization of each algorithm on a single CMAP scan.
2.  A workflow for batch processing an entire dataset.

The sample data in this repository is included in Data/CA-EDM 9, but any CMAP scan can be ran given a MEM file describing it, or a group of CMAP scans in an MEF file.

## Directory Structure

The repository is organized to be straightforward. The analysis notebook expects this structure to run correctly:

```
CMAP-MUNE-Analysis/
├── Data/
│   └── CA-EDM 9/
│       ├── CA-EDM-MSF2 APB-1 9.MEF
│       ├── CA-EDM-MSF2 APB-2 9.MEF
│       └── ... (All other sample .MEM files)
│
├── Scripts/
│   ├── DataHandler.py     # For loading .MEM and .MEF files
│   ├── STEPIX.py
│   ├── CDIX.py
│   └── Stairfit.py
│
├── Results/
│   └── (This directory is created automatically to store outputs)
│
├── analysis_notebook.ipynb        # Main user interface
├── requirements.txt
└── README.md
```

## Setup and Installation

To get started, clone the repository and install the required dependencies.

```bash
pip install -r requirements.txt
```

## Usage

The primary way to use this repository is through the Jupyter Notebook.

1.  **Launch Jupyter Notebook or JupyterLab:**
    ```bash
    jupyter notebook
    ```
2.  **Open `analysis_notebook.ipynb`**.
3.  **Run the cells sequentially.** The notebook is documented and will guide you through loading data, running the algorithms, and saving the results.

## Expected Outputs

After running the full notebook and the figure generation scripts, the `Results/` directory will contain:

*   **`batch_results_stepix.csv`**: Collated STEPIX results.
*   **`batch_results_cdix.csv`**: Collated CDIX results.
*   **`batch_results_stairfit.csv`**: Collated Stairfit MUNE results.