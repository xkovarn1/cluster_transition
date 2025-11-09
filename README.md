# Cluster Transition Analysis

`cluster_transition.py` provides a full workflow for **spatiotemporal clustering and transition analysis** using a multi-decadal 1 km gridded database of continental-scale spring onset products.  
It uses **xarray**, **rioxarray**, **scikit-learn**, and **matplotlib** to cluster derived indices and analyze how spatial patterns evolve across time.

The links to the data:
* Dataset description: https://www.nature.com/articles/s41597-024-03710-5
* Dataset repo: https://data.4tu.nl/datasets/aca56a60-8fcc-45b5-b817-50b68d4b5c63

The workflow expects the data to be stored as follows:

    ├── data/
    │   └── european_indices/
    │       └── Damage_index/
    |       └── Last_freeze/
    |       └── Leaf/
    |       └── Bloom/

---

## Features

- **Load and preprocess GeoTIFF datasets** by index and year  
- **K-Means clustering (MiniBatch)** on spatial-temporal data  
- **Cluster alignment** between time periods using the **Hungarian algorithm**  
- **Transition matrix** computation between cluster states  
- **Visualization utilities**:
  - Cluster maps per time period  
  - Transition maps between periods  
  - Change detection masks  
- **Cluster and transition summaries**:
  - Mean, median, and standard deviation per cluster  
  - Area changes, inflow/outflow, and retention statistics  

---


## Notebook

The `spring_indices.ipynb` demonstrates the usage of the above described functions.
